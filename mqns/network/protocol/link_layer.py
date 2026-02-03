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
from typing import Literal, TypedDict, cast, override

from mqns.entity.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.network import TimingPhase, TimingPhaseEvent
from mqns.network.protocol.event import (
    LinkArchSuccessEvent,
    ManageActiveChannels,
    QubitDecoheredEvent,
    QubitEntangledEvent,
    QubitReleasedEvent,
)
from mqns.utils import json_encodable, log, rng


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


@json_encodable
class LinkLayerCounters:
    def __init__(self):
        self.n_etg = 0
        """how many entanglements generated as the primary node"""
        self.n_attempts = 0
        """how many attempts made for successful entanglements"""
        self.n_decoh = 0
        """how many qubits decohered"""

    def increment_n_etg(self, attempts: int) -> None:
        self.n_etg += 1
        self.n_attempts += attempts


class LinkLayer(Application[QNode]):
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
        """
        Constructor.

        Args:
            attempt_rate: max entanglement attempts per second (default: 1e6) (currently ineffective).
            alpha_db_per_km: fiber attenuation loss in dB/km (default: 0.2).
            eta_s: source efficiency (default: 1.0).
            eta_d: detector efficiency (default: 1.0).
            frequency: entanglement source frequency in Hz (default: 80e6).
            tau_0: local operation delay in seconds for emitting and absorbing photon (default: 0.0).
            init_fidelity: fidelity of generated entangled pairs (default: 0.99).

        """
        super().__init__()

        self.attempt_interval = 1 / attempt_rate
        """Minimum interval spaced out between attempts (currently ineffective)."""
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

        self.active_channels: dict[tuple[QuantumChannel, int | None], tuple[QNode, int]] = {}
        """
        Active quantum channels and paths where own node is the primary (on the left side).
        Key is qchannel and optional path_id.
        Value is [0] neighbor QNode [1] insertion count.

        When qubits were assigned to qchannels but not allocated to specific path_ids,
        such as in MuxSchemeDynamicEpr, path_id would be None, but LinkLayer would still
        receive one ManageActiveChannels start event per qchannel+path combination.
        The insertion count keeps track of how many start events were received
        for the qchannel that are not yet canceled by corresponding stop events.

        When qubits were allocated to specific path_ids, insertion count should always be 1.
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

        self.cnt = LinkLayerCounters()
        """
        Counters.
        """

        # event handlers
        self.add_handler(self.handle_sync_phase, TimingPhaseEvent)
        self.add_handler(self.RecvClassicPacketHandler, RecvClassicPacket)
        self.add_handler(self.handle_manage_active_channels, ManageActiveChannels)
        self.add_handler(self.handle_success_entangle, LinkArchSuccessEvent)
        self.add_handler(self.handle_decoh_rel, (QubitDecoheredEvent, QubitReleasedEvent))

    @override
    def install(self, node):
        self._application_install(node, QNode)
        self.memory = self.node.memory
        """Quantum memory of the node."""
        self.epr_type = self.node.network.epr_type
        """Network-wide entanglement type."""

    def handle_sync_phase(self, event: TimingPhaseEvent):
        """
        Handle timing phase signals, only used in SYNC timing mode.

        Upon entering EXTERNAL phase:

        1. Clear existing memory qubits.
        2. Run all active channels from reservation step.

        Upon entering INTERNAL phase: do nothing.
        """
        if event.phase == TimingPhase.EXTERNAL:
            self.memory.clear()
            for (qchannel, path_id), (neighbor, _) in self.active_channels.items():
                self.run_active_channel(qchannel, path_id, neighbor)

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
        if event.start:
            self.add_active_channel(event.qchannel, event.path_id, event.neighbor)
        else:
            self.remove_active_channel(event.qchannel, event.path_id, event.neighbor)
        return True

    def add_active_channel(self, qchannel: QuantumChannel, path_id: int | None, neighbor: QNode):
        key = (qchannel, path_id)
        _, n = self.active_channels.get(key, (neighbor, 0))
        n += 1
        self.active_channels[key] = (neighbor, n)

        if n > 1:
            assert path_id is None
            return

        # link_arch.set() may be called multiple times in several situations:
        # - qchannel is activated, deactivated, and re-activated.
        # - qchannel is activated from both sides due to shared paths in opposite directions.
        # Nevertheless, we assume every LinkLayer instance has the same parameters,
        # so that the parameters saved into LinkArch are the same every time.
        qchannel.link_arch.set(
            length=qchannel.length,
            alpha=self.alpha_db_per_km,
            eta_s=self.eta_s,
            eta_d=self.eta_d,
            reset_time=self.reset_time,
            tau_l=qchannel.delay.calculate(),
            tau_0=self.tau_0,
            epr_type=self.epr_type,
            init_fidelity=self.init_fidelity,
        )

        log.debug(
            f"{self.node}: add qchannel {qchannel} with {neighbor} on path {path_id}, link arch {qchannel.link_arch.name}"
        )

        if self.node.timing.is_async():
            self.run_active_channel(qchannel, path_id, neighbor)

    def remove_active_channel(self, qchannel: QuantumChannel, path_id: int | None, neighbor: QNode):
        key = (qchannel, path_id)
        _, n = self.active_channels[key]
        n -= 1

        if n == 0:
            del self.active_channels[key]
            log.debug(f"{self.node}: remove qchannel {qchannel} with {neighbor} on path {path_id}")
        else:
            self.active_channels[key] = (neighbor, n)

    def run_active_channel(self, qchannel: QuantumChannel, path_id: int | None, next_hop: QNode):
        """
        Start EPR generation over the given quantum channel and the specified path_id.

        Args:
            qchannel: The quantum channel over which entanglement is to be attempted.
            path_id: The path_id to restrict attempts to path-allocated qubits only.
                    Needed in multipath where a channel may be activated while not all qubits have been allocated to paths.
            next_hop: The neighboring node with which to initiate the negotiation.

        Notes:
            - Qubits assigned to memory are retrieved using the channel's name.
            - Qubit reservations are spaced out in time using a fixed `attempt_rate`.

        """
        qubits = list(self.memory.find(lambda *_: True, qchannel=qchannel))
        log.debug(f"{self.node}: {qchannel.name} has assigned qubits: {qubits}")
        for qb, data in qubits:
            if qb.path_id != path_id or qb.state != QubitState.RAW:
                continue
            assert qb.active is None
            assert data is None, f"{self.node}: qubit {qb} has data {data}"
            self.start_reservation(next_hop, qchannel, qb)

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
              Key format: `<node1>_<node2>_[<path_id>]_<local_qubit_addr>`
            - The reservation is communicated via a classical message using the `RESERVE_QUBIT` command.
        """

        key = uuid.uuid4().hex
        assert key not in self.pending_init_reservation
        qubit.state, qubit.active = QubitState.ACTIVE, key
        self.pending_init_reservation[key] = (qchannel, next_hop, qubit)
        log.debug(f"{self.node}: start reservation key={key} dst={next_hop} addr={qubit.addr} path={qubit.path_id}")

        msg: ReserveMsg = {"cmd": "RESERVE_QUBIT", "path_id": qubit.path_id, "key": key}
        cchannel = self.node.get_cchannel(next_hop)
        cchannel.send(ClassicPacket(msg, src=self.node, dest=next_hop), next_hop=next_hop)

    def handle_reserve_req(self, msg: ReserveMsg, cchannel: ClassicChannel):
        """
        Handle `RESERVE_QUBIT` control message sent by the initiating node to request a memory qubit reservation.

        1. If an available memory qubit is found, it is reserved (marked active using the given key).
        2. A `RESERVE_QUBIT_OK` response is sent back to confirm the reservation.
        3. If no available qubit is found, the request is enqueued for future retry (FIFO).
        """
        from_node = cchannel.find_peer(self.node)
        assert type(from_node) is QNode
        qchannel = self.node.get_qchannel(from_node)
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
        qubit, _ = next(
            self.memory.find(
                lambda q, v: (
                    v is None  # currently unoccupied
                    and not q.active  # not part of an active reservation
                    and q.path_id == req.path_id  # allocated to the path_id, if MuxScheme uses path_id
                ),
                qchannel=req.qchannel,  # assigned to the quantum channel
            ),
            (None, None),
        )
        if qubit is None:
            return False

        log.debug(f"{self.node}: accept reservation key={req.key} src={req.from_node} addr={qubit.addr} path={qubit.path_id}")
        qubit.state = QubitState.ACTIVE  # cannot go directly from RAW to RESERVED
        qubit.state, qubit.active = QubitState.RESERVED, req.key
        msg: ReserveMsg = {"cmd": "RESERVE_QUBIT_OK", "path_id": req.path_id, "key": req.key}
        req.cchannel.send(ClassicPacket(msg, src=self.node, dest=req.from_node), next_hop=req.from_node)
        return True

    def handle_reserve_res(self, msg: ReserveMsg):
        """
        Handle `RESERVE_QUBIT_OK` control messages received as a response to a reservation request.

        1. Trigger the entanglement generation process using the reserved memory qubit.
        """
        key = msg["key"]
        (qchannel, next_hop, qubit) = self.pending_init_reservation.pop(key)
        assert qubit.active == key
        qubit.state = QubitState.RESERVED
        self.generate_entanglement(qchannel, next_hop, qubit)

    def generate_entanglement(self, qchannel: QuantumChannel, next_hop: QNode, qubit: MemoryQubit):
        """
        Schedule a successful entanglement attempt using skip-ahead sampling.

        Args:
            qchannel: The quantum channel over which entanglement is to be generated.
            next_hop: The neighboring node with which the entanglement is attempted.
            qubit: The memory qubit used for this attempt.
        """
        # Calculate which attempt would succeed.
        k = rng.geometric(qchannel.link_arch.success_prob)

        # Calculate when would the k-th attempt (1-based) succeed.
        # TODO space out EPRs on a qchannel by attempt_interval or qchannel.bandwidth
        epr, t_notify_a, t_notify_b = qchannel.link_arch.make_epr(
            k, self.simulator.tc, key=qubit.active, src=self.node, dst=next_hop
        )

        # If the network uses SYNC timing mode but the successful attempt would exceed the current EXTERNAL phase,
        # the EPR would not arrive in time, and therefore is not scheduled.
        if not self.node.timing.is_external(max(t_notify_a, t_notify_b)):
            log.debug(
                f"{self.node}: skip prepare EPR {epr.name} key={epr.key} dst={epr.dst} attempts={k} "
                f"notify-times={t_notify_a},{t_notify_b} reason=beyond-external-phase"
            )
            return

        # If the network uses ASYNC timing mode or the successful attempt can complete within the current EXTERNAL phase,
        # schedule the EPR arrival on both nodes via LinkArchSuccessEvents.
        log.debug(
            f"{self.node}: prepare EPR {epr.name} key={epr.key} dst={epr.dst} attempts={k} "
            f"notify-times={t_notify_a},{t_notify_b}"
        )

        self.simulator.add_event(LinkArchSuccessEvent(self.node, epr, t=t_notify_a, by=self, attempts=k))
        self.simulator.add_event(LinkArchSuccessEvent(next_hop, epr, t=t_notify_b, by=self, attempts=k))

    def handle_success_entangle(self, event: LinkArchSuccessEvent):
        assert self.node.timing.is_external()

        epr = event.epr
        neighbor, is_primary = (epr.dst, True) if epr.src == self.node else (epr.src, False)
        assert neighbor is not None
        if is_primary:
            self.cnt.increment_n_etg(event.attempts)

        log.debug(f"{self.node}: got half-EPR {epr.name} key={epr.key} {'dst' if is_primary else 'src'}={neighbor}")
        assert epr.decohere_time > self.simulator.tc

        qubit = self.memory.write(epr.key, epr)
        if qubit is None:
            raise Exception(f"{self.node}: Failed to store EPR {epr.name}")

        qubit.state = QubitState.ENTANGLED0
        self.simulator.add_event(QubitEntangledEvent(self.node, neighbor, qubit, t=self.simulator.tc, by=self))

    def handle_decoh_rel(self, event: QubitDecoheredEvent | QubitReleasedEvent) -> bool:
        qubit = event.qubit
        is_decoh = type(event) is QubitDecoheredEvent
        if is_decoh:
            self.cnt.n_decoh += 1
            log.debug(f"{self.node}: qubit decohered addr={qubit.addr} old-key={qubit.active}")
        else:
            log.debug(f"{self.node}: qubit released addr={qubit.addr} old-key={qubit.active}")

        qubit.state, qubit.active = QubitState.RAW, None

        assert qubit.qchannel is not None
        ac = self.active_channels.get((qubit.qchannel, qubit.path_id))
        if ac is None:  # this node is not the EPR initiator
            # Check deferred reservation requests and attempt to fulfill the reservation.
            # Only the first (oldest) request in the FIFO is processed per call.
            if self.fifo_reservation_req and self.try_accept_reservation(self.fifo_reservation_req[0]):
                self.fifo_reservation_req.popleft()
        else:  # this node is the EPR initiator
            next_hop, _ = ac
            if self.node.timing.is_async():
                self.start_reservation(next_hop, qubit.qchannel, qubit)
            elif is_decoh:
                raise Exception(f"{self.node}: UNEXPECTED -> (t_ext + t_int) too short")
        return True
