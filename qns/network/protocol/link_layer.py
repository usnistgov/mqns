#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import uuid
from collections import deque
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

import numpy as np

from qns.entity.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from qns.entity.memory import MemoryQubit, QuantumMemory
from qns.entity.node import Application, Node, QNode
from qns.entity.qchannel import QuantumChannel
from qns.models.epr import WernerStateEntanglement
from qns.network import SignalTypeEnum, TimingModeEnum
from qns.network.protocol.event import (
    LinkArchSuccessEvent,
    ManageActiveChannels,
    QubitDecoheredEvent,
    QubitEntangledEvent,
    QubitReleasedEvent,
    TypeEnum,
)
from qns.simulator import Simulator, func_to_event
from qns.utils import log


class ReserveMsg(TypedDict):
    cmd: Literal["RESERVE_QUBIT", "RESERVE_QUBIT_OK"]
    path_id: int | None
    key: str


@dataclass
class ReservationRequest:
    key: str
    path_id: int | None
    cchannel: ClassicChannel
    from_node: QNode
    qchannel: QuantumChannel


class LinkLayer(Application):
    """
    Network function for creating elementary entanglements over qchannels.
    It equips a QNode and is activated from the forwarding function (e.g., ProactiveForwarder).
    """

    def __init__(
        self,
        *,
        attempt_rate: float = 1e6,
        alpha_db_per_km: float = 0.2,
        eta_s: float = 1.0,
        eta_d: float = 1.0,
        frequency: float = 80e6,
        tau_0: float = 0.0,
        init_fidelity: float = 0.99,
    ):
        """This constructor sets up the entanglement generation layer of a quantum node with key hardware parameters.
        It also initializes data structures for managing quantum channels, entanglement attempts,
        and synchronization.

        Args:
            attempt_rate: max entanglement attempts per second (default: 1e6).
            alpha_db_per_km: fiber attenuation loss in dB/km (default: 0.2).
            eta_s: source efficiency (default: 1.0).
            eta_d: detector efficiency (default: 1.0).
            frequency: entanglement source frequency in Hz (default: 80e6).
            tau_0: local operation delay in seconds for emitting and absorbing photon (default: 0.0).
            init_fidelity: fidelity of generated entangled pairs (default: 0.99).

        """

        super().__init__()

        self.attempt_interval = 1 / attempt_rate
        """Minimum interval spaced out between attempts."""
        self.alpha_db_per_km = alpha_db_per_km
        """Fiber attenuation loss in dB/km."""
        self.eta_s = eta_s
        """Source efficiency between 0 and 1."""
        self.eta_d = eta_d
        """Detector efficiency between 0 and 1."""
        self.reset_time = 1 / frequency
        """Minimum time between two consecutive photon excitations/absorptions."""
        self.tau_0 = tau_0
        """Local operation delay in seconds."""
        self.init_fidelity = init_fidelity
        """Fidelity of generated entangled pairs."""

        self.own: QNode
        """Quantum node that owns this LinkLayer."""
        self.memory: QuantumMemory
        """Quantum memory of the node."""

        self.active_channels: dict[tuple[str, int | None], tuple[QuantumChannel, QNode]] = {}
        """
        Table of active quantum channels and paths.
        Key is qchannel name and optional path_id.
        Value is the qchannel and remote QNode.
        """
        self.pending_init_reservation: dict[str, tuple[QuantumChannel, QNode, MemoryQubit]] = {}
        """
        Table of pending reservations for which RESERVE_QUBIT is sent but RESERVE_QUBIT_OK has not arrived.
        Key is reservation key.
        Value is the qchannel, next hop QNode, local qubit.
        """
        self.fifo_reservation_req = deque[ReservationRequest]()
        """
        FIFO queue of reservation requests awaiting for memory qubits.
        """

        self.etg_count = 0
        """Counter of generated entanglements."""
        self.decoh_count = 0
        """Counter of decohered qubits never swapped."""

        # event handlers
        self.add_handler(self.RecvClassicPacketHandler, RecvClassicPacket)
        self.add_handler(self.handle_manage_active_channels, ManageActiveChannels)
        self.add_handler(self.handle_success_entangle, LinkArchSuccessEvent)
        self.add_handler(self.handle_decoh_rel, [QubitDecoheredEvent, QubitReleasedEvent])

    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        self.own = self.get_node(node_type=QNode)
        self.memory = self.own.get_memory()

    def handle_sync_signal(self, signal_type: SignalTypeEnum):
        """Handles timing synchronization signals for SYNC mode (not very reliable at this time)."""
        if signal_type == SignalTypeEnum.EXTERNAL:
            # clear all qubits and retry all active_channels until INTERNAL signal
            self.memory.clear()
            for (_, path_id), (qchannel, next_hop) in self.active_channels.items():
                self.run_active_channel(qchannel, next_hop, path_id)

    def RecvClassicPacketHandler(self, event: RecvClassicPacket) -> bool:
        msg = event.packet.get()
        if not (isinstance(msg, dict) and "cmd" in msg):
            return False

        match msg["cmd"]:
            case "RESERVE_QUBIT":
                self.handle_reserve_req(cast(ReserveMsg, msg), event.cchannel)
                return True
            case "RESERVE_QUBIT_OK":
                self.handle_reserve_res(cast(ReserveMsg, msg))
                return True
            case _:
                return False

    def handle_manage_active_channels(self, event: ManageActiveChannels) -> bool:
        """Handle ManageActiveChannels event from forwarder."""
        qchannel = self.own.get_qchannel(event.neighbor)
        if event.type == TypeEnum.ADD:
            self.add_active_channel(qchannel, event.neighbor, event.path_id)
        else:
            self.remove_active_channel(qchannel, event.neighbor, event.path_id)
        return True

    def add_active_channel(self, qchannel: QuantumChannel, neighbor: QNode, path_id: int | None):
        key = (qchannel.name, path_id)
        if key in self.active_channels:  # ignore duplicate add
            return

        log.debug(f"{self.own}: add qchannel {qchannel} with {neighbor} on path {path_id}, link arch {qchannel.link_arch.name}")
        self.active_channels[key] = (qchannel, neighbor)
        if self.own.timing_mode == TimingModeEnum.ASYNC:
            self.run_active_channel(qchannel, neighbor, path_id)

    def remove_active_channel(self, qchannel: QuantumChannel, neighbor: QNode, path_id: int | None):
        log.debug(f"{self.own}: remove qchannel {qchannel} with {neighbor} on path {path_id}")
        self.active_channels.pop((qchannel.name, path_id), None)

    def run_active_channel(self, qchannel: QuantumChannel, next_hop: QNode, path_id: int | None):
        """
        Start EPR generation over the given quantum channel and the specified next-hop.
        It performs qubit reservation, and for each available qubit, an EPR creation event is scheduled
        with a staggered delay based on EPR generation sampling.

        Args:
            qchannel: The quantum channel over which entanglement is to be attempted.
            next_hop: The neighboring node with which to initiate the negotiation.
            path_id: The path_id to restrict attempts to path-allocated qubits only.
                    Needed in multipath where a channel may be activated while not all qubits have been allocated to paths.

        Raises:
            Exception: If a qubit assigned to the channel is unexpectedly already associated
                   with a `QuantumModel`, indicating a logic error or memory mismanagement.

        Notes:
            - Qubits assigned to memory are retrieved using the channel's name.
            - Qubit reservations are spaced out in time using a fixed `attempt_rate`.

        """
        simulator = self.simulator
        qubits = self.memory.get_channel_qubits(ch_name=qchannel.name)
        log.debug(f"{self.own}: {qchannel.name} has assigned qubits: {qubits}")
        for i, (qb, data) in enumerate(qubits):
            if qb.path_id != path_id:
                continue
            if qb.active is None:
                if data is not None:
                    raise Exception(f"{self.own}: qubit has data {data}")
                simulator.add_event(
                    func_to_event(
                        simulator.tc + i * self.attempt_interval, self.start_reservation, next_hop, qchannel, qb, by=self
                    )
                )

    def start_reservation(self, next_hop: QNode, qchannel: QuantumChannel, qubit: MemoryQubit):
        """
        Start the exchange with neighbor node for reserving a qubit for entanglement
        generation over a specified quantum channel. It performs the following steps:

        1. Construct a random reservation `key`.
        2. Mark the qubit as active using the reservation key.
        3. Store reservation metadata in `pending_init_reservation`.
        4. Send a classical message to the next hop to request qubit reservation.

        Args:
            next_hop: The neighboring node with which the reservation is to be made.
            qchannel: The quantum channel used for entanglement.
            qubit: The memory qubit to reserve.

        Raises:
            Exception: If a reservation has already been initiated for the same key,
                   or if no classical channel to the destination node is found.

        Notes:
            - The `key` uniquely identifies the reservation context.
            Key format: <node1>_<node2>_[<path_id>]_<local_qubit_addr>
            - The reservation is communicated via a classical message using the `RESERVE_QUBIT` command.
        """

        key = uuid.uuid4().hex
        assert key not in self.pending_init_reservation
        log.debug(f"{self.own}: start reservation | key = {key} | path = {qubit.path_id}")
        qubit.active = key
        self.pending_init_reservation[key] = (qchannel, next_hop, qubit)

        msg: ReserveMsg = {"cmd": "RESERVE_QUBIT", "path_id": qubit.path_id, "key": key}
        cchannel = self.own.get_cchannel(next_hop)
        cchannel.send(ClassicPacket(msg, src=self.own, dest=next_hop), next_hop=next_hop)

    def handle_reserve_req(self, msg: ReserveMsg, cchannel: ClassicChannel):
        """
        Handle `RESERVE_QUBIT` control message sent by the initiating node to request a memory qubit reservation.
        1. If an available memory qubit is found, it is reserved (marked active using the given key).
        2. A `RESERVE_QUBIT_OK` response is sent back to confirm the reservation.
        3. If no available qubit is found, the request is enqueued for future retry (FIFO).
        """
        from_node = cchannel.find_peer(self.own)
        assert isinstance(from_node, QNode)
        qchannel = self.own.get_qchannel(from_node)
        req = ReservationRequest(msg["key"], msg["path_id"], cchannel, from_node, qchannel)
        if not self.try_accept_reservation(req):
            self.fifo_reservation_req.append(req)

    def try_accept_reservation(self, req: ReservationRequest) -> bool:
        """
        Accept a reservation if a qubit is available.

        Returns:
            True if the reservation is accepted and `RESERVE_QUBIT_OK` is sent.
            False if the reservation is not accepted.

        Notes: Caller is responsible for managing `fifo_reservation_req` queue.
        """
        avail_qubits = self.memory.search_available_qubits(ch_name=req.qchannel.name, path_id=req.path_id)
        if not avail_qubits:
            return False

        avail_qubits[0].active = req.key
        msg: ReserveMsg = {"cmd": "RESERVE_QUBIT_OK", "path_id": req.path_id, "key": req.key}
        req.cchannel.send(ClassicPacket(msg, src=self.own, dest=req.from_node), next_hop=req.from_node)
        return True

    def handle_reserve_res(self, msg: ReserveMsg):
        """
        Handle `RESERVE_QUBIT_OK` control messages received as a response to a reservation request.
        1. Trigger the entanglement generation process using the reserved memory qubit.
        """
        key = msg["key"]
        (qchannel, next_hop, qubit) = self.pending_init_reservation.pop(key)
        assert qubit.active == key
        self.generate_entanglement(qchannel=qchannel, next_hop=next_hop, qubit=qubit)

    def generate_entanglement(self, qchannel: QuantumChannel, next_hop: QNode, qubit: MemoryQubit):
        """Schedule a successful entanglement attempt using skip-ahead sampling.
        A `do_successful_attempt` event is scheduled to handle the result of this attempt.

        It performs the following checks and steps:
            - Ensures the memory's decoherence time is sufficient for the channel length.
            - Computes the time of the next successful attempt and number of skipped trials.
            - Schedules a successful entanglement event at the computed time.

        Args:
            qchannel (QuantumChannel): The quantum channel over which entanglement is to be generated.
            next_hop (QNode): The neighboring node with which the entanglement is attempted.
            address (int): The address of the memory qubit used for this attempt.
            key (str): A unique identifier for this entanglement reservation/attempt.

        """
        simulator = self.simulator

        # Calculate the time until the successful attempt.
        # Then, the last tau of the the successful attempt is simulated by sending the EPR
        # from primary node to secondary node.
        p = qchannel.link_arch.success_prob(
            length=qchannel.length, alpha=self.alpha_db_per_km, eta_s=self.eta_s, eta_d=self.eta_d
        )
        k = np.random.geometric(p)  # k-th attempt will succeed

        tau_l = qchannel.delay_model.calculate()  # time to send photon/message one way
        d_epr_creation, d_notify_primary, d_notify_secondary = qchannel.link_arch.delays(
            k,
            reset_time=self.reset_time,
            tau_l=tau_l,
            tau_0=self.tau_0,
        )
        t_epr_creation = simulator.tc + d_epr_creation
        # TODO investigate why some procedures crash without adding 1 time slot
        t_notify_primary = simulator.tc + d_notify_primary + simulator.time(time_slot=1)
        t_notify_secondary = simulator.tc + d_notify_secondary

        epr = WernerStateEntanglement(fidelity=self.init_fidelity, name=uuid.uuid4().hex)
        epr.src = self.own
        epr.dst = next_hop
        epr.attempts = k
        epr.key = qubit.active
        epr.path_id = qubit.path_id
        epr.creation_time = t_epr_creation

        log.debug(
            f"{self.own}: prepare EPR {epr.name} key={epr.key} dst={epr.dst} attempts={k} "
            f"times={t_epr_creation},{t_notify_primary},{t_notify_secondary}"
        )

        simulator.add_event(LinkArchSuccessEvent(self.own, epr, t=t_notify_primary, by=self))
        simulator.add_event(LinkArchSuccessEvent(next_hop, epr, t=t_notify_secondary, by=self))

    def handle_success_entangle(self, event: LinkArchSuccessEvent):
        if self.own.timing_mode == TimingModeEnum.SYNC and self.sync_current_phase != SignalTypeEnum.EXTERNAL:
            log.debug(f"{self.own}: EXT phase is over -> stop attempts")
            return

        simulator = self.simulator
        epr = event.epr
        neighbor, is_primary = (epr.dst, True) if epr.src == self.own else (epr.src, False)
        assert neighbor is not None
        if is_primary:
            self.etg_count += 1

        log.debug(f"{self.own}: got half-EPR {epr.name} key={epr.key} {'secondary' if is_primary else 'primary'}={neighbor}")
        assert epr.decoherence_time is None or epr.decoherence_time > self.simulator.tc

        qubit = self.memory.write(epr, path_id=epr.path_id, key=epr.key)
        if qubit is None:
            raise Exception(f"{self.own}: Failed to store EPR {epr.name}")

        qubit.purif_rounds = 0
        qubit.fsm.to_entangled()
        simulator.add_event(QubitEntangledEvent(self.own, neighbor, qubit, t=simulator.tc, by=self))

    def handle_decoh_rel(self, event: QubitDecoheredEvent | QubitReleasedEvent) -> bool:
        is_decoh = isinstance(event, QubitDecoheredEvent)
        if is_decoh:
            self.decoh_count += 1
        qubit = event.qubit
        assert qubit.qchannel is not None
        ac = self.active_channels.get((qubit.qchannel.name, qubit.path_id))
        if ac is None:  # this node is not the EPR initiator
            qubit.active = None

            # Check deferred reservation requests and attempt to fulfill the reservation.
            # Only the first (oldest) request in the FIFO is processed per call.
            if self.fifo_reservation_req and self.try_accept_reservation(self.fifo_reservation_req[0]):
                self.fifo_reservation_req.popleft()
        else:  # this node is the EPR initiator
            qchannel, next_hop = ac
            if self.own.timing_mode == TimingModeEnum.ASYNC:
                self.start_reservation(next_hop, qchannel, qubit)
            elif is_decoh and self.own.timing_mode == TimingModeEnum.SYNC:
                raise Exception(f"{self.own}: UNEXPECTED -> (t_ext + t_int) too short")
        return True
