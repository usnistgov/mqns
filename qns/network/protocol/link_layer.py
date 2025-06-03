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
from typing import Optional

import numpy as np

from qns.entity.cchannel.cchannel import ClassicPacket, RecvClassicPacket
from qns.entity.memory.memory import QuantumMemory
from qns.entity.memory.memory_qubit import MemoryQubit
from qns.entity.node.app import Application
from qns.entity.node.node import Node
from qns.entity.node.qnode import QNode
from qns.entity.qchannel.qchannel import QuantumChannel, RecvQubitPacket
from qns.models.epr import WernerStateEntanglement
from qns.network import SignalTypeEnum, TimingModeEnum
from qns.simulator.event import Event, func_to_event
from qns.simulator.simulator import Simulator
from qns.simulator.ts import Time
from qns.utils import log


class LinkLayer(Application):
    """This is the network function reponsible for creating elementary entanglements over qchannels.
    It equips a QNode and is activated from the forwarding function (e.g., ProactiveForwarder).
    """

    def __init__(self,
                 attempt_rate: int = 1e6,
                 alpha_db_per_km: float = 0.2,
                 eta_d: float = 1.0,
                 eta_s: float = 1.0,
                 frequency: int = 80e6,
                 init_fidelity: int = 0.99,
                 light_speed_kms = 2 * 10**5):
        """This constructor sets up the entanglement generation layer of a quantum node with key hardware parameters.
        It also initializes data structures for managing quantum channels, entanglement attempts,
        and synchronization.

        Parameters
        ----------
            attempt_rate (int): Max entanglement attempts per second (default: 1e6).
            alpha_db_per_km (float): Fiber loss in dB/km (default: 0.2).
            eta_d (float): Detector efficiency (default: 1.0).
            eta_s (float): Source efficiency (default: 1.0).
            frequency (int): Entanglement source frequency (default: 80e6).
            init_fidelity (int): Fidelity of generated entangled pairs (default: 0.99).
            light_speed_kms (int): Speed of light in fiber in km/s (default: 2e5).

        """
        from qns.network.protocol.proactive_forwarder import ProactiveForwarder
        super().__init__()

        self.alpha_db_per_km = alpha_db_per_km
        self.eta_s = eta_s
        self.eta_d = eta_d
        self.frequency = frequency
        self.init_fidelity = init_fidelity
        self.attempt_rate = attempt_rate
        self.light_speed_kms = light_speed_kms

        self.own: QNode                    # Quantum node this LinkLayer equips
        self.memory: QuantumMemory         # Quantum memory of the node
        self.forwarder: ProactiveForwarder # Forwarder function of the node

        # stores the qchannels activated by the forwarding function at path installation
        self.active_channels = {}

        self.pending_init_reservation = {}          # stores reservation requests sent by this node
        self.fifo_reservation_req = []              # stores received reservations requests awaiting for qubits

        self.etg_count = 0                          # counts number of generated entanglements
        self.decoh_count = 0                        # counts number of decohered qubits never swapped

        self.sync_current_phase = SignalTypeEnum.EXTERNAL      # for SYNC and LSYNC timing modes

        # In LSYNC mode: stores the qchannels that have all their qubits waiting for the next EXTERNAL phase
        self.waiting_channels = {}
        # In LSYNC mode: stores the qubits waiting for the next EXTERNAL phase
        self.waiting_qubits = set()

        # handlers for extenral events
        self.add_handler(self.RecvQubitHandler, [RecvQubitPacket])
        self.add_handler(self.RecvClassicPacketHandler, [RecvClassicPacket])


    # called at initialization of the node
    def install(self, node: QNode, simulator: Simulator):
        from qns.network.protocol.proactive_forwarder import ProactiveForwarder

        super().install(node, simulator)
        self.own = self.get_node(node_type=QNode)
        self.memory = self.own.get_memory()
        fwd_fns = self.own.get_apps(ProactiveForwarder)
        if fwd_fns:
            self.forwarder = fwd_fns[0]
        else:
            raise Exception("No forwarder found")

    def RecvQubitHandler(self, node: QNode, event: Event):
        self.receive_quit(event)

    def RecvClassicPacketHandler(self, node: Node, event: Event):
        if event.packet.get()["cmd"] in ["RESERVE_QUBIT", "RESERVE_QUBIT_OK"]:
            self.handle_reservation(event)

    def handle_active_channel(self, qchannel: QuantumChannel, next_hop: QNode):
        """This method starts EPR generation over the given quantum channel and the specified next-hop.
        It performs qubit reservation, and for each available qubit, an EPR creation event is scheduled
        with a staggered delay based on EPR generation samping.

        Args:
            qchannel (QuantumChannel): The quantum channel over which entanglement is to be attempted.
            next_hop (QNode): The neighboring node with which to initiate the negotiation.

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
            if data is None:
                t = simulator.tc + i * 1 / self.attempt_rate
                event = func_to_event(t, self.start_reservation, by=self,
                                      next_hop=next_hop, qchannel=qchannel,
                                      qubit=qb, path_id=qb.path_id)
                simulator.add_event(event)
            else:
                raise Exception(f"{self.own}: --> PROBLEM {data}")


    def start_reservation(self, next_hop: Node, qchannel: QuantumChannel,
                          qubit: MemoryQubit, path_id: Optional[int] = None):
        """This method starts the exchange with neighbor node for reserving a qubit for entanglement
        generation over a specified quantum channel. It performs the following steps:

        - Constructs a reservation `key` using the current node, next hop, optional path ID, and qubit address.
        - Verifies that a reservation has not already been initiated for this key.
        - Marks the qubit as active using the reservation key.
        - Stores reservation metadata in `pending_init_reservation`.
        - Retrieves the classical channel to the next hop.
        - Sends a classical message to the next hop to request qubit reservation.

        Args:
            next_hop (Node): The neighboring node with which the reservation is to be made.
            qchannel (QuantumChannel): The quantum channel used for entanglement.
            qubit (MemoryQubit): The memory qubit to reserve.
            path_id (Optional[int]): Optional identifier for the entanglement path.

        Raises:
            Exception: If a reservation has already been initiated for the same key,
                   or if no classical channel to the destination node is found.

        Notes:
            - The `key` uniquely identifies the reservation context.
            - The reservation is communicated via a classical message using the `RESERVE_QUBIT` command.

        """
        key = self.own.name +"_"+ next_hop.name
        if path_id is not None:
            key = key+"_"+str(path_id)
        key = key+"_"+str(qubit.addr)

        if key in self.pending_init_reservation:
            raise Exception(f"{self.own}: reservation already started for {key}")
            return

        log.debug(f"{self.own}: start reservation with key={key}")
        qubit.active = key
        self.pending_init_reservation[key] = (qchannel, next_hop, qubit.addr)
        cchannel = self.own.get_cchannel(next_hop)
        classic_packet = ClassicPacket(msg={"cmd": "RESERVE_QUBIT", "path_id": path_id, "key": key},
                                       src=self.own, dest=next_hop)
        cchannel.send(classic_packet, next_hop=next_hop)


    def generate_entanglement(self, qchannel: QuantumChannel, next_hop: Node,
                              address: int, key: str):
        """Schedule a successful entanglement attempt using skip-ahead sampling.
        A `do_successful_attempt` event is scheduled to handle the result of this attempt.

        It performs the following checks and steps:
            - Verifies that the quantum channel is currently active.
            - Ensures the memory's decoherence time is sufficient for the channel length.
            - Computes the time of the next successful attempt and number of skipped trials.
            - Schedules a successful entanglement event at the computed time.

        Args:
            qchannel (QuantumChannel): The quantum channel over which entanglement is to be generated.
            next_hop (Node): The neighboring node with which the entanglement is attempted.
            address (int): The address of the memory qubit used for this attempt.
            key (str): A unique identifier for this entanglement reservation/attempt.

        Raises:
            Exception:
                - If the quantum channel is not currently marked as active.
                - If the channel is too long for successful entanglement before decoherence.

        """
        simulator = self.simulator
        if qchannel.name not in self.active_channels:
            raise Exception(f"{self.own}: Qchannel not active")
            return

        t_mem = 1 / self.memory.decoherence_rate
        if qchannel.length >= (2 * self.light_speed_kms * t_mem):
            raise Exception("Qchannel too long for entanglement attempt.")

        succ_attempt_time, attempts = self._skip_ahead_entanglement(qchannel.length)
        t_event = simulator.tc + succ_attempt_time
        event = func_to_event(t_event, self.do_successful_attempt, by=self, qchannel=qchannel,
                              next_hop=next_hop, address=address, attempts=attempts, key=key)
        simulator.add_event(event)


    def do_successful_attempt(self, qchannel: QuantumChannel, next_hop: Node,
                              address, attempts: int, key: str):
        """This method is invoked after a scheduled successful entanglement attempt. It:
            - Generates a new EPR pair between the current node and the next hop.
            - Stores the EPR locally at the specified memory address, accounting for the qubit initialization time.
            - Sends the remote half of the EPR over the quantum channel to the next hop.
            - Notifies the forwarder accounting for the protocol delay.

        Args:
            qchannel (QuantumChannel): The quantum channel used for transmission.
            next_hop (Node): The neighboring node with which entanglement is established.
            address (int): The memory address at which to store the local half of the EPR.
            attempts (int): The number of entanglement attempts before this success.
            key (str): An identifier of the qubit reservation for the neighbor node's qubit.

        Notes:
            - The quantum channel is assumed to have no photon loss in this step since it executes only the sucessful attempt.

        """
        epr = WernerStateEntanglement(fidelity=self.init_fidelity, name=uuid.uuid4().hex)
        # qubit init at 2tau and we are at 6tau
        epr.creation_time = self.simulator.tc - Time(sec=4*qchannel.delay_model.calculate())
        epr.src = self.own
        epr.dst = next_hop
        epr.attempts = attempts
        epr.key = key

        local_qubit = self.memory.write(qm=epr, address=address)

        if not local_qubit:
            raise Exception(f"{self.own}: (sender) Do succ EPR -> memory full, key ({key})")

        epr.path_id = local_qubit.path_id
        qchannel.send(epr, next_hop)    # no drop
        self.etg_count+=1
        self.notify_entangled_qubit(neighbor=next_hop, qubit=local_qubit, \
            delay=qchannel.delay_model.calculate())   # wait 1tau to notify

    def receive_quit(self, packet: RecvQubitPacket):
        """This method is called when a quantum channel delivers an entangled qubit (half of an EPR pair)
        to the local node. It performs the following:

            - Extracts the `WernerStateEntanglement` (the received qubit) from the packet.
            - Attempts to store the received qubit in memory accounting for qubit initialization time.
            - Notifies the forwarder that a qubit has been successfully entangled.

        Args:
            packet (RecvQubitPacket): The packet containing the received qubit.

        Notes:
            - In `SYNC` timing mode, the node must be in the `EXTERNAL` signal phase to accept entangled qubits.

        """
        if self.own.timing_mode == TimingModeEnum.SYNC and self.sync_current_phase != SignalTypeEnum.EXTERNAL:
            log.debug(f"{self.own}: EXT phase is over -> stop attempts")
            return

        qchannel: QuantumChannel = packet.qchannel
        from_node: Node = qchannel.node_list[0] \
            if qchannel.node_list[1] == self.own else qchannel.node_list[1]

        epr: WernerStateEntanglement = packet.qubit

        log.debug(f"{self.own}: recv half-EPR {epr.name} from {from_node} | reservation key {epr.key}")

        if epr.decoherence_time <= self.simulator.tc:
            raise Exception(f"{self.own}: Decoherence time already passed | {epr}")

        # qubit init at 2tau and we are at 7*tau
        local_qubit = self.memory.write(qm=epr, path_id=epr.path_id, key=epr.key)

        if local_qubit is None:
            raise Exception(f"{self.own}: Failed to store rcvd EPR due to full memory")

        self.notify_entangled_qubit(neighbor=from_node, qubit=local_qubit)

    def notify_entangled_qubit(self, neighbor: QNode, qubit: MemoryQubit, delay: float = 0):
        """Schedule an event to notify the forwarder about a new entangled qubit
        """
        simulator = self.simulator
        from qns.network.protocol.event import QubitEntangledEvent
        qubit.fsm.to_entangled()
        t = simulator.tc + delay
        event = QubitEntangledEvent(forwarder=self.forwarder, neighbor=neighbor, qubit=qubit, t=t, by=self)
        simulator.add_event(event)


    def handle_event(self, event: Event) -> None:
        """Handles the following external events:
        - `ManageActiveChannels` from the forwarder to start/stop generating EPRs with a given neighbor.
        - `QubitDecoheredEvent` from memory informing that an entangled qubit has decohered and can be re-entangled.
        - `QubitReleasedEvent` from the forwarder informing that an entangled qubit has released and can be re-entangled.
        """
        from qns.network.protocol.event import ManageActiveChannels, QubitDecoheredEvent, QubitReleasedEvent, TypeEnum
        if isinstance(event, ManageActiveChannels):
            log.debug(f"{self.own}: start qchannel with {event.neighbor}")
            qchannel = self.own.get_qchannel(event.neighbor)
            if event.type == TypeEnum.ADD:
                if qchannel.name not in self.active_channels:
                    self.active_channels[qchannel.name] = (qchannel, event.neighbor)
                    if self.own.timing_mode == TimingModeEnum.ASYNC:
                        self.handle_active_channel(qchannel, event.neighbor)
                    elif self.own.timing_mode == TimingModeEnum.LSYNC:     # LSYNC
                        self.waiting_channels[qchannel.name] = (qchannel, event.neighbor)
                else:
                    raise Exception("Qchannel already handled")
            else:
                self.active_channels.pop(qchannel.name, "Not Found")
        elif isinstance(event, QubitDecoheredEvent):
            self.decoh_count+=1
            # check if this node is the EPR initiator of the qchannel associated with the memory of this qubit
            if event.qubit.qchannel.name:
                if event.qubit.qchannel.name in self.active_channels:
                    if self.own.timing_mode == TimingModeEnum.LSYNC:
                        raise Exception(f"{self.own}: UNEXPECTED -> t_slot too short")
                    if self.own.timing_mode == TimingModeEnum.SYNC:
                        raise Exception(f"{self.own}: UNEXPECTED -> (t_ext + t_int) too short")
                    qchannel, next_hop = self.active_channels[event.qubit.qchannel.name]
                    self.start_reservation(next_hop=next_hop, qchannel=qchannel,
                                           qubit=event.qubit, path_id=event.qubit.path_id)
                else:
                    event.qubit.active = None
                    self.check_reservation_req()
            else:
                raise Exception("TODO")
        elif isinstance(event, QubitReleasedEvent):
            # check if this node is the EPR initiator of the qchannel associated with the memory of this qubit
            if event.qubit.qchannel.name:
                if event.qubit.qchannel.name in self.active_channels:     # i.e., this node is primary
                    qchannel, next_hop = self.active_channels[event.qubit.qchannel.name]
                    if self.own.timing_mode == TimingModeEnum.ASYNC:
                        self.start_reservation(next_hop=next_hop, qchannel=qchannel,
                                               qubit=event.qubit, path_id=event.qubit.path_id)
                    elif self.own.timing_mode == TimingModeEnum.LSYNC:    # LSYNC
                        entry = (qchannel, next_hop, event.qubit.qchannel.name, event.qubit.addr)
                        self.waiting_qubits.add(entry)
                else:
                    event.qubit.active = None
                    self.check_reservation_req()
            else:
                raise Exception("TODO")


    def handle_reservation(self, packet: RecvClassicPacket):
        """Handle classical control messages related to qubit reservation.

        1. `RESERVE_QUBIT`: Sent by the initiating node to request a memory qubit reservation.
            - If an available memory qubit is found, it is reserved (marked active using the given key).
            - A `RESERVE_QUBIT_OK` response is sent back to confirm the reservation.
            - If no available qubit is found, the request is enqueued for future retry (FIFO).

        2. `RESERVE_QUBIT_OK` - Received as a response to a reservation request.
            - Triggers the entanglement generation process using the reserved memory qubit.

        Args:
            packet (RecvClassicPacket): The packet containing the control message and associated classical channel.

        Raises:
            Exception: If no corresponding quantum channel exists between the current node and the sender.

        Notes:
            - The `key` uniquely identifies a reservation and is used to properly store the generated EPR.
            - This method is used everytime an initialor node wants to generate a new entanglement.
            - FIFO buffering allows for deferred reservation attempts when memory is temporarily unavailable.

        """
        msg = packet.packet.get()
        cchannel = packet.cchannel
        from_node: QNode = cchannel.node_list[0] if cchannel.node_list[1] == self.own else cchannel.node_list[1]
        qchannel = self.own.get_qchannel(from_node)

        cmd = msg["cmd"]
        path_id = msg["path_id"]
        key = msg["key"]
        if cmd == "RESERVE_QUBIT":
            log.debug(f"{self.own}: rcvd RESERVE_QUBIT {key}")
            avail_qubits = self.memory.search_available_qubits(path_id=path_id)
            if avail_qubits:
                log.debug(f"{self.own}: direct found available qubit for {key}")
                avail_qubits[0].active = key
                classic_packet = ClassicPacket(msg={"cmd": "RESERVE_QUBIT_OK", "path_id": path_id, "key": key},
                                               src=self.own, dest=from_node)
                cchannel.send(classic_packet, next_hop=from_node)
            else:
                log.debug(f"{self.own}: didn't find available qubit for {key}")
                self.fifo_reservation_req.append((key, path_id, cchannel, from_node))
        elif cmd == "RESERVE_QUBIT_OK":
            log.debug(f"{self.own}: returned qubit available with {key}")
            (qchannel, next_hop, address) = self.pending_init_reservation[key]
            self.generate_entanglement(qchannel=qchannel, next_hop=next_hop, address=address, key=key)
            self.pending_init_reservation.pop(key, None)

    def check_reservation_req(self):
        """This method handles reservation requests that were previously deferred due to a lack
        of available memory qubits. It checks the front of the FIFO queue and attempts to
        fulfill the reservation:
            - If a free memory qubit is found that matches the requested path ID, it is marked as active.
            - A `RESERVE_QUBIT_OK` classical message is sent to the requesting node to confirm the allocation.
            - The request is then removed from the queue.

        Notes:
            - Only the first (oldest) request in the FIFO is processed per call.
            - This method is triggered when a qubit is released or decohered.

        """
        if self.fifo_reservation_req:
            (key, path_id, cchannel, from_node) = self.fifo_reservation_req[0]
            log.debug(f"{self.own}: handle pending negoc {key}")
            avail_qubits = self.memory.search_available_qubits(path_id=path_id)
            if avail_qubits:
                log.debug(f"{self.own}: found available qubit for {key}")
                avail_qubits[0].active = key
                classic_packet = ClassicPacket(msg={"cmd": "RESERVE_QUBIT_OK", "path_id": path_id, "key": key},
                                               src=self.own, dest=from_node)
                cchannel.send(classic_packet, next_hop=from_node)
                self.fifo_reservation_req.pop(0)


    def handle_sync_signal(self, signal_type: SignalTypeEnum):
        """Handles timing synchronization signals for SYNC and LSYNC modes (not very reliable at this time).
        """
        log.debug(f"{self.own}:[{self.own.timing_mode}] TIMING SIGNAL <{signal_type}>")
        if self.own.timing_mode == TimingModeEnum.LSYNC and signal_type == SignalTypeEnum.EXTERNAL_START:
            # clear all qubits and retry all active_channels until INTERNAL signal
            self.memory.clear()
            for channel_name, (qchannel, next_hop) in self.active_channels.items():
                self.handle_active_channel(qchannel, next_hop)
        elif self.own.timing_mode == TimingModeEnum.SYNC:
            self.sync_current_phase = signal_type
            if signal_type == SignalTypeEnum.EXTERNAL:
                # clear all qubits and retry all active_channels until INTERNAL signal
                self.memory.clear()
                for channel_name, (qchannel, next_hop) in self.active_channels.items():
                    self.handle_active_channel(qchannel, next_hop)


    def _loss_based_success_prob(self, link_length_km):
        """Compute success probability from fiber loss model for heralded entanglement."""
        p_bsa = 0.5
        p_fiber = 10 ** (-self.alpha_db_per_km * link_length_km / 10)
        p = p_bsa * (self.eta_s**2) * (self.eta_d**2) * p_fiber
        return p

    def _skip_ahead_entanglement(self, link_length_km: float):
        reset_time = 1 / self.frequency
        tau = link_length_km / self.light_speed_kms

        # probability assumes that each attempt has always 2-rounds
        p = self._loss_based_success_prob(link_length_km)
        k = np.random.geometric(p)     # k-th attempt will succeed

        attempt_duration = max(5.5*tau, reset_time)    # includes 1*tau between attempts (not rounds)

        # calculate time right before the successful attempt
        # the last 1-tau of the successful attempt will be executed
        # substract 2tau consumed for reservation (just to alignt with sequence)
        t_success = ((k-1) * attempt_duration) + (4*tau)
        return t_success, k
