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

import copy
from typing import TYPE_CHECKING

from qns.entity.cchannel import ClassicPacket, RecvClassicPacket
from qns.entity.memory import QuantumMemory
from qns.entity.memory.memory_qubit import MemoryQubit, QubitState
from qns.entity.node import Application, Controller, Node, QNode
from qns.models.epr import WernerStateEntanglement
from qns.network import QuantumNetwork, SignalTypeEnum, TimingModeEnum
from qns.network.protocol.event import QubitReleasedEvent
from qns.network.protocol.fib import FIBEntry, ForwardingInformationBase
from qns.simulator import Event, Simulator
from qns.utils import log

if TYPE_CHECKING:
    from qns.network.protocol.link_layer import LinkLayer


class ProactiveForwarder(Application):
    """ProactiveForwarder is the forwarder of QNodes and receives routing instructions from the controller.
    It implements the forwarding phase (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller. Purification will be moved to a sepeare network function.
    """

    def __init__(self, ps: float = 1.0):
        """This constructor sets up a node's entanglement forwarding logic in a quantum network.
        It configures the swapping success probability and preparing internal
        state for managing memory, routing instructions (via FIB), synchronization,
        and classical communication handling.

        Parameters
        ----------
            ps (float): Probability of successful entanglement swapping (default: 1.0).

        """
        super().__init__()

        self.ps = ps

        self.net: QuantumNetwork  # Quantum Network instance
        self.own: QNode  # Quantum node this Forwarder equips
        self.memory: QuantumMemory  # Quantum memory of the node

        self.fib: ForwardingInformationBase = ForwardingInformationBase()  # FIB structure
        self.link_layer: "LinkLayer"  # Reference to the network function responsible for generating elementary EPRs

        # for SNYC mode
        self.sync_current_phase = SignalTypeEnum.INTERNAL
        self.waiting_qubits = []  # stores the qubits waiting for the INTERNAL phase (SYNC mode)

        # handler for classical packets
        self.add_handler(self.RecvClassicPacketHandler, RecvClassicPacket)

        self.parallel_swappings = {}  # structure to manage potential parallel swappings

        self.e2e_count = 0  # counts number of e2e generated EPRs
        self.fidelity = 0.0  # stores fidelity of generated EPRs

    # called at initialization of the node
    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        self.own = self.get_node(node_type=QNode)
        self.memory = self.own.get_memory()
        self.net = self.own.network

        from qns.network.protocol.link_layer import LinkLayer

        self.link_layer = self.own.get_app(LinkLayer)

    # receives a classical packet and dispatches it as control or signaling
    def RecvClassicPacketHandler(self, node: Node, event: RecvClassicPacket):
        if isinstance(event.packet.src, Controller):
            self.handle_control(event)
        elif isinstance(event.packet.src, QNode):
            self.handle_signaling(event)
        else:
            log.warn(f"Unexpected event from entity type: {type(event.packet.src)}")

    def handle_control(self, packet: RecvClassicPacket):
        """Processes a classical packet containing routing instructions from the controller.
        Determines left/right neighbors from the route, identifies corresponding quantum channels,
        and allocates qubits based on the multiplexing vector (for the buffer-space mode).
        Updates the FIB with path, swapping, and purification info, and triggers EPR generation via the
        link layer on the outgoing channel.
        No path allocation for qubits (i.e., no `m_v` vector) means statistical mux is required, and it is currently ignored.

        Parameters
        ----------
            packet (RecvClassicPacket): Classical packet containing routing instructions.

        """
        simulator = self.simulator
        msg = packet.packet.get()
        log.debug(f"{self.own.name}: routing instructions: {msg}")

        path_id = msg["path_id"]
        instructions = msg["instructions"]

        left_neighbor = None
        right_neighbor = None
        ln = ""
        rn = ""

        # get left and right nodes from route vector:
        if self.own.name in instructions["route"]:
            i = instructions["route"].index(self.own.name)
            ln, rn = (
                instructions["route"][i - 1] if i > 0 else None,
                instructions["route"][i + 1] if i < len(instructions["route"]) - 1 else None,
            )
        else:
            raise Exception(f"Node {self.own.name} not found in route vector {instructions['route']}")

        # use left and right nodes to get qchannels
        if ln:
            left_neighbor = self.net.get_node(ln)
            self.own.get_qchannel(left_neighbor)  # ensure qchannel exists

        if rn:
            right_neighbor = self.own.network.get_node(rn)
            self.own.get_qchannel(right_neighbor)  # ensure qchannel exists

        # use mux info to allocate qubits in each memory, keep qubit addresses
        left_qubits = []
        right_qubits = []

        if instructions["m_v"]:
            num_left, num_next = self.compute_qubit_allocation(instructions["route"], instructions["m_v"], self.own.name)
            if num_left:
                if num_left <= self.memory.count_unallocated_qubits():
                    for i in range(num_left):
                        left_qubits.append(self.memory.allocate(path_id=path_id))
                else:
                    raise Exception("Not enough qubits for left qchannel allocation")
            if num_next:
                if num_next <= self.memory.count_unallocated_qubits():
                    for i in range(num_next):
                        right_qubits.append(self.memory.allocate(path_id=path_id))
                else:
                    raise Exception("Not enough qubits for right qchannel allocation")
            log.debug(f"Allocated qubits: left = {left_qubits} | right = {right_qubits}")
        else:
            log.debug(f"{self.own}: No m_v provided -> Statistical multiplexing not supported yet")
            return

        # populate FIB
        self.fib.add_entry(
            replace=True,
            path_id=path_id,
            path_vector=instructions["route"],
            swap_sequence=instructions["swap"],
            purification_scheme=instructions["purif"],
            qubit_addresses=[],
        )

        # call network function responsible for generating EPRs on right qchannel
        if right_neighbor:
            from qns.network.protocol.event import ManageActiveChannels, TypeEnum

            t = simulator.tc
            ll_request = ManageActiveChannels(
                link_layer=self.link_layer, neighbor=right_neighbor, type=TypeEnum.ADD, t=t, by=self
            )
            simulator.add_event(ll_request)

        # TODO: remove path
        # t = simulator.tc
        # ll_request = ManageActiveChannels(link_layer=self.link_layer, neighbor=right_hop,
        #                                           type=TypeEnum.REMOVE, t=t, by=self)
        # simulator.add_event(ll_request)

    def handle_signaling(self, packet: RecvClassicPacket):
        """Parses and dispatches classical signaling commands (e.g., for entanglement
        swapping or purification coordination) to the appropriate handler based on
        message type. Determines the sender node via the classical channel metadata.

        Parameters
        ----------
            packet (RecvClassicPacket): Classical packet carrying the signaling message.

        """
        dest = packet.packet.dest
        msg = packet.packet.get()
        cchannel = packet.cchannel
        from_node = cchannel.node_list[0] if cchannel.node_list[1] == self.own else cchannel.node_list[1]
        assert isinstance(from_node, QNode)
        assert isinstance(dest, QNode)

        # TODO: do msg forwarding here

        if msg["cmd"] == "SWAP_UPDATE":
            self.handle_swap_update(msg, from_node, dest)
        elif msg["cmd"] == "PURIF_SOLICIT":
            self.handle_purif_solicit(msg, from_node, dest)
        elif msg["cmd"] == "PURIF_RESPONSE":
            self.handle_purif_response(msg, from_node, dest)

    def handle_swap_update(self, msg: dict, from_node: QNode, dest_node: QNode):
        """Processes an SWAP_UPDATE signaling message from a neighboring node, updating local
        qubit state, releasing decohered pairs, or forwarding the update along the path.
        Handles both sequential and parallel swap scenarios, updates quantum memory with
        new EPRs when valid, and updates the qubit state.

        Parameters
        ----------
            msg (Dict): The SWAP_UPDATE message.
            from_node (QNode): The node that sent the message.
            dest_node (QNode): The destination node for this signaling message.

        """
        simulator = self.simulator
        path_id = msg["path_id"]
        if self.own.timing_mode == TimingModeEnum.SYNC and self.sync_current_phase != SignalTypeEnum.INTERNAL:
            log.debug(f"{self.own}: INT phase is over -> stop swaps")
            return

        fib_entry = self.fib.get_entry(path_id)
        if not fib_entry:
            raise Exception(f"{self.own}: FIB entry not found for path {path_id}")

        route = fib_entry["path_vector"]
        swap_sequence = fib_entry["swap_sequence"]
        sender_idx = route.index(msg["swapping_node"])
        sender_rank = swap_sequence[sender_idx]
        own_idx = route.index(self.own.name)
        own_rank = swap_sequence[own_idx]

        # destination means:
        # - the node needs to update its local qubit wrt remote node (partner)
        # - this etg **side** becomes ready to purify/swap
        if dest_node.name == self.own.name:
            if own_rank > sender_rank:  # this node didn't swap yet
                qubit = self.get_memory_qubit(msg["epr"])
                if qubit:
                    # swap failed or oldest pair decohered -> release qubit
                    if msg["new_epr"] is None or msg["new_epr"].decoherence_time <= simulator.tc:
                        if msg["new_epr"]:
                            log.debug(f"{self.own}: NEW EPR {msg['new_epr']} decohered during SU transmissions")
                        self.memory.read(address=qubit.addr)
                        qubit.fsm.to_release()
                        event = QubitReleasedEvent(link_layer=self.link_layer, qubit=qubit, t=simulator.tc, by=self)
                        simulator.add_event(event)
                    else:  # update old EPR with new EPR (fidelity and partner)
                        updated = self.memory.update(old_qm=msg["epr"], new_qm=msg["new_epr"])
                        if not updated:
                            log.debug(f"### {self.own}: VERIFY -> EPR update {updated}")
                        if updated and self.eval_qubit_eligibility(fib_entry, msg["partner"]):
                            qubit.fsm.to_purif()
                            partner = self.own.network.get_node(msg["partner"])
                            self.purif(qubit, fib_entry, partner)
                else:  # epr decohered -> release qubit
                    log.debug(f"{self.own}: EPR {msg['epr']} decohered during SU transmissions")
            elif own_rank == sender_rank:  # the two nodes may have swapped
                # log.debug(f"### {self.own}: rcvd SU from same-rank node {msg['new_epr']}")
                qubit = self.get_memory_qubit(msg["epr"])
                if qubit:  # there was no parallel swap
                    self.parallel_swappings.pop(msg["epr"], None)  # clean parallel_swappings
                    if msg["new_epr"] is None or msg["new_epr"].decoherence_time <= simulator.tc:
                        self.memory.read(address=qubit.addr)
                        qubit.fsm.to_release()
                        event = QubitReleasedEvent(link_layer=self.link_layer, qubit=qubit, t=simulator.tc, by=self)
                        simulator.add_event(event)
                    else:  # update old EPR with new EPR (fidelity and partner)
                        updated = self.memory.update(old_qm=msg["epr"], new_qm=msg["new_epr"])
                        if not updated:
                            log.debug(f"### {self.own}: VERIFY -> EPR update {updated}")
                elif msg["epr"] in self.parallel_swappings:
                    (shared_epr, other_epr, my_new_epr) = self.parallel_swappings[msg["epr"]]
                    if msg["new_epr"] is None or msg["new_epr"].decoherence_time <= simulator.tc:
                        if other_epr.dst == self.own:
                            destination = other_epr.src
                            partner = shared_epr.dst
                        else:
                            destination = other_epr.dst
                            partner = shared_epr.src
                        fwd_msg = {
                            "cmd": "SWAP_UPDATE",
                            "path_id": msg["path_id"],
                            "swapping_node": msg["swapping_node"],
                            "partner": partner.name,
                            "epr": my_new_epr.name,
                            "new_epr": None,
                            "fwd": True,
                        }
                        self.send_msg(dest=destination, msg=fwd_msg, route=fib_entry["path_vector"], delay=True)
                        self.parallel_swappings.pop(msg["epr"], None)
                    else:  # a neighbor successfully swapped in parallel with this node
                        new_epr = msg["new_epr"]  # is the epr from neighbor swap
                        merged_epr = new_epr.swapping(epr=other_epr)  # merge the two swaps (physically already happened)
                        if other_epr.dst == self.own:
                            if merged_epr is not None:
                                merged_epr.src = other_epr.src
                                merged_epr.dst = new_epr.dst
                            partner = new_epr.dst.name
                            destination = other_epr.src
                        else:
                            if merged_epr is not None:
                                merged_epr.src = new_epr.src
                                merged_epr.dst = other_epr.dst
                            partner = new_epr.src.name
                            destination = other_epr.dst
                        fwd_msg = {
                            "cmd": "SWAP_UPDATE",
                            "path_id": msg["path_id"],
                            "swapping_node": msg["swapping_node"],
                            "partner": partner,
                            "epr": my_new_epr.name,
                            "new_epr": merged_epr,
                            "fwd": True,
                        }
                        self.send_msg(dest=destination, msg=fwd_msg, route=fib_entry["path_vector"], delay=True)
                        self.parallel_swappings.pop(msg["epr"], None)

                        # update parallel swappings for next potential cases:
                        p_idx = route.index(partner)
                        p_rank = swap_sequence[p_idx]
                        if (own_rank == p_rank) and (merged_epr is not None):
                            self.parallel_swappings[new_epr.name] = (new_epr, other_epr, merged_epr)
                else:
                    log.debug(f"### {self.own}: EPR {msg['epr']} decohered after swapping [parallel]")
            else:
                log.debug(f"### {self.own}: VERIFY -> rcvd SU from higher-rank node")
        # node is not destination of this SU: forward message
        elif own_rank <= sender_rank:
            msg_copy = copy.deepcopy(msg)
            log.debug(f"{self.own}: FWD SWAP_UPDATE")
            msg_copy["fwd"] = True
            self.send_msg(dest=dest_node, msg=msg_copy, route=fib_entry["path_vector"])
        else:
            log.debug(f"### {self.own}: VERIFY -> not the swapping dest and did not swap")

    def eval_qubit_eligibility(self, fib_entry: FIBEntry, partner: str) -> bool:
        """Evaluate if a qubit is eligible for purification.
        Compares the local node's swap rank to its partner's in the given path.
        A qubit is eligible for purification if the partner's swap rank is greater
        than or equal to the local node's rank.

        Parameters
        ----------
            fib_entry (Dict): FIB entry containing the path and swap sequence.
            partner (str): Name of the partner node to compare against.

        Returns
        -------
            bool: True if eligible for swap/purification, False otherwise.

        """
        route = fib_entry["path_vector"]
        swap_sequence = fib_entry["swap_sequence"]
        partner_idx = route.index(partner)
        partner_rank = swap_sequence[partner_idx]
        own_idx = route.index(self.own.name)
        own_rank = swap_sequence[own_idx]

        # If partner rank is higher or equal -> go to PURIF
        if partner_rank >= own_rank:
            return True
        return False

    def purif(self, qubit: MemoryQubit, fib_entry: FIBEntry, partner: QNode):
        """Called when a qubit transitions to the PURIF state.
        Determines the segment in which the qubit is entangled and number of required purification rounds from the FIB.
        If the required rounds are completed, the qubit becomes eligible. Otherwise, the node evaluates
        whether it is the initiator for the purification (i.e., primary). If so, it searches for an auxiliary
        qubit to use, consumes the auxiliary, updates qubit states, and sends a PURIF_SOLICIT message
        to the partner node.

        Parameters
        ----------
            qubit (MemoryQubit): The memory qubit at PURIF state.
            fib_entry (Dict): FIB entry containing routing and purification instructions.
            partner (QNode): The node with which the qubit shares an EPR.

        """
        simulator = self.simulator
        assert qubit.qchannel is not None
        # TODO: make this controllable
        # # for isolated links -> consume immediatly
        # _, qm = self.memory.read(address=qubit.addr, must=True)
        # qubit.fsm.to_release()
        # log.debug(f"{self.own}: consume entanglement: <{qubit.addr}> {qm.src.name} - {qm.dst.name}")
        # event = QubitReleasedEvent(link_layer=self.link_layer, qubit=qubit, e2e=self.own.name=='S',
        #                            t=simulator.tc, by=self)
        # simulator.add_event(event)

        route = fib_entry["path_vector"]
        swap_sequence = fib_entry["swap_sequence"]
        partner_idx = route.index(partner.name)
        partner_rank = swap_sequence[partner_idx]
        own_idx = route.index(self.own.name)
        own_rank = swap_sequence[own_idx]

        segment_name = f"{self.own.name}-{partner.name}" if own_idx < partner_idx else f"{partner.name}-{self.own.name}"
        purif_scheme = fib_entry["purification_scheme"]

        if segment_name not in purif_scheme:
            log.debug(f"{self.own}: no purif instructions for segment {segment_name}")
            qubit.fsm.to_eligible()
            self.eligible(qubit, fib_entry)
            return

        purif_rounds = purif_scheme[segment_name]
        log.debug(f"{self.own}: segment {segment_name} (qubit {qubit.addr}) needs {purif_rounds} purif rounds")

        if qubit.purif_rounds == purif_rounds:
            log.debug(f"{self.own}: {purif_rounds} purif rounds done for qubit {qubit.addr}")
            qubit.purif_rounds = 0
            qubit.fsm.to_eligible()
            self.eligible(qubit, fib_entry)
            return

        primary = False
        if own_rank < partner_rank:
            primary = True
        elif own_rank == partner_rank:
            primary = own_idx < partner_idx

        log.debug(f"{self.own}: is primary node {primary}")
        if not primary:
            return

        # primary node:
        res = self.select_purif_qubit(
            exc_address=qubit.addr,
            partner=partner.name,
            qchannel=qubit.qchannel.name,
            path_id=fib_entry["path_id"],
            purif_rounds=qubit.purif_rounds,
        )
        if res:
            log.debug(f"{self.own}: available qubit for purif {res} -> start purif")

            # sets the fidelity for the partner at this time
            _, epr = self.memory.read(address=qubit.addr, destructive=False, must=True)
            assert isinstance(epr, WernerStateEntanglement)
            assert epr.name is not None

            # consume and release other_qubit
            # this sets the fidelity for the partner at this time
            other_qubit, other_epr = self.memory.read(address=res.addr, must=True)
            assert isinstance(other_epr, WernerStateEntanglement)
            assert other_epr.name is not None

            other_qubit.fsm.to_release()
            ev = QubitReleasedEvent(link_layer=self.link_layer, qubit=other_qubit, t=simulator.tc, by=self)
            simulator.add_event(ev)

            qubit.fsm.to_pending()  # epr to keep goes to pending

            # send purif_solicit to partner
            msg = {
                "cmd": "PURIF_SOLICIT",
                "path_id": fib_entry["path_id"],
                "purif_node": self.own.name,
                "partner": partner.name,
                "epr": epr.name,
                "measure_epr": other_epr.name,
                "round": qubit.purif_rounds,
            }
            self.send_msg(dest=partner, msg=msg, route=route)
        else:
            log.debug(f"{self.own}: no other EPR is available for purif")

    def handle_purif_solicit(self, msg: dict, from_node: QNode, dest_node: QNode):
        """Processes a PURIF_SOLICIT message from a partner node as part of the purification protocol.
        Retrieves the target and auxiliary qubits from memory, verifies their states, and attempts
        purification. If successful, updates the EPR and sends a PURIF_RESPONSE with result=True;
        otherwise, marks both qubits for release and replies with result=False. If the current node
        is not the message destination, the message is forwarded along the route.

        Parameters
        ----------
            msg (Dict): Message containing purification parameters and EPR names.
            from_node (QNode): Node that sent the PURIF_SOLICIT message.
            dest_node (QNode): Intended destination node for the message.

        """
        simulator = self.simulator
        path_id = msg["path_id"]
        fib_entry = self.fib.get_entry(path_id)
        if not fib_entry:
            raise Exception(f"{self.own}: FIB entry not found for path {path_id}")

        if dest_node.name == self.own.name:
            qubit, epr = self.memory.read(
                key=msg["epr"], destructive=False, must=True
            )  # gets the same fidelity as primary node
            meas_qubit, meas_epr = self.memory.read(key=msg["measure_epr"], must=True)  # gets the same fidelity as primary node
            # TODO: verify (should be decohered) in case either EPR is not found in memory
            assert isinstance(epr, WernerStateEntanglement)
            assert epr.name is not None
            assert isinstance(meas_epr, WernerStateEntanglement)

            if qubit.fsm.state != QubitState.ENTANGLED or meas_qubit.fsm.state != QubitState.ENTANGLED:
                log.debug(f"{self.own}: qubit={qubit.fsm.state}, meas_qubit={meas_qubit.fsm.state}")
                raise Exception(f"{self.own}: qubits not in ELIGIBLE states -> not suppoted yet")

            # if qubits in ELIGIBLE -> do purif, release measured qubit, update purified qubit-pair, reply
            dest = self.own.network.get_node(msg["purif_node"])
            resp_msg = {
                "cmd": "PURIF_RESPONSE",
                "path_id": msg["path_id"],
                "purif_node": msg["purif_node"],
                "partner": self.own.name,
                "epr": epr.name,
                "measure_epr": meas_epr.name,
                "round": msg["round"],
            }

            if epr.purify(meas_epr):  # purif succ
                resp_msg["result"] = True
                self.memory.update(old_qm=epr.name, new_qm=epr)
                self.send_msg(dest=dest, msg=resp_msg, route=fib_entry["path_vector"])
            else:  # purif failed
                resp_msg["result"] = False
                self.send_msg(dest=dest, msg=resp_msg, route=fib_entry["path_vector"])

                self.memory.read(address=qubit.addr)  # desctructive reading
                qubit.fsm.to_release()
                ev = QubitReleasedEvent(link_layer=self.link_layer, qubit=qubit, t=simulator.tc, by=self)
                simulator.add_event(ev)

            # measured qubit already released
            meas_qubit.fsm.to_release()
            ev = QubitReleasedEvent(link_layer=self.link_layer, qubit=meas_qubit, t=simulator.tc, by=self)
            simulator.add_event(ev)

            # TODO: if qubits in PURIF -> do purif, release consumed qubit,
            # update pair + increment rounds (if succ, else release), reply
        else:  # node is not destination: forward message
            self.send_msg(dest=dest_node, msg=msg, route=fib_entry["path_vector"])

    def handle_purif_response(self, msg: dict, from_node: QNode, dest_node: QNode):
        """Handles a PURIF_RESPONSE message indicating the outcome of a purification attempt.
        If the current node is the destination and purification succeeded, the EPR is updated,
        the qubit's purification round counter is incremented, and the qubit may re-enter the
        purification process. If purification failed, the qubit is released. If the current node
        is not the destination, the message is forwarded along the path.

        Parameters
        ----------
            msg (Dict): Response message containing the result and identifiers of the purified EPRs.
            from_node (QNode): Node that sent the PURIF_RESPONSE message.
            dest_node (QNode): Intended destination node for the message.

        """
        simulator = self.simulator
        path_id = msg["path_id"]
        fib_entry = self.fib.get_entry(path_id)
        if not fib_entry:
            raise Exception(f"{self.own}: FIB entry not found for path {path_id}")

        if dest_node.name == self.own.name:
            qubit, epr = self.memory.get(key=msg["epr"], must=True)
            # TODO: verify (should be decohered) in case EPR is not found in memory
            if msg["result"]:  # purif succeeded
                assert isinstance(epr, WernerStateEntanglement)
                assert epr.name is not None
                self.memory.update(old_qm=epr.name, new_qm=epr)
                qubit.purif_rounds += 1
                qubit.fsm.to_purif()
                partner = self.own.network.get_node(msg["partner"])
                self.purif(qubit, fib_entry, partner)
            else:  # purif failed -> release qubit
                self.memory.read(address=qubit.addr)  # desctructive reading
                qubit.fsm.to_release()
                ev = QubitReleasedEvent(link_layer=self.link_layer, qubit=qubit, t=simulator.tc, by=self)
                simulator.add_event(ev)
        else:  # node is not destination: forward message
            self.send_msg(dest=dest_node, msg=msg, route=fib_entry["path_vector"])

    def eligible(self, qubit: MemoryQubit, fib_entry: FIBEntry):
        """Called when a qubit enters the ELIGIBLE state, either to attempt entanglement swapping
        (if the node is intermediate) or to finalize consumption (if the node is an end node).
        Intermediate nodes look for a matching eligible qubit to perform swapping, generate a
        new EPR if successful, and notify adjacent nodes with SWAP_UPDATE messages. End nodes consume the EPR.

        Parameters
        ----------
            qubit (MemoryQubit): The qubit that became eligible.
            fib_entry (Dict): FIB entry containing path and swap sequence.

        """
        simulator = self.simulator
        assert qubit.qchannel is not None
        if self.own.timing_mode == TimingModeEnum.SYNC and self.sync_current_phase != SignalTypeEnum.INTERNAL:
            log.debug(f"{self.own}: INT phase is over -> stop swaps")
            return

        swap_sequence = fib_entry["swap_sequence"]
        route = fib_entry["path_vector"]
        own_idx = route.index(self.own.name)
        if own_idx > 0 and own_idx < len(route) - 1:  # intermediate node
            # look for another eligible qubit
            res = self.select_eligible_qubit(exc_qchannel=qubit.qchannel.name, path_id=fib_entry["path_id"])
            if res:  # do swapping
                # Read both qubits to set current fidelity
                other_qubit, other_epr = self.memory.read(address=res.addr, must=True)
                this_qubit, this_epr = self.memory.read(address=qubit.addr, must=True)
                assert isinstance(this_epr, WernerStateEntanglement)
                assert isinstance(other_epr, WernerStateEntanglement)

                # order eprs and prev/next nodes
                if this_epr.dst == self.own:
                    prev_partner = this_epr.src
                    prev_epr = this_epr
                    next_partner = other_epr.dst
                    next_epr = other_epr

                    prev_qubit = this_qubit
                    next_qubit = other_qubit
                elif this_epr.src == self.own:
                    prev_partner = other_epr.src
                    prev_epr = other_epr
                    next_partner = this_epr.dst
                    next_epr = this_epr

                    prev_qubit = other_qubit
                    next_qubit = this_qubit
                else:
                    raise Exception(f"Unexpected: swapping EPRs {this_epr} x {other_epr}")

                assert prev_partner is not None
                assert next_partner is not None

                # if elementary epr -> assign ch_index
                if not prev_epr.orig_eprs:
                    prev_epr.ch_index = own_idx - 1
                if not next_epr.orig_eprs:
                    next_epr.ch_index = own_idx

                new_epr = this_epr.swapping(epr=other_epr, ps=self.ps)
                log.debug(f"{self.own}: SWAP {'SUCC' if new_epr else 'FAILED'} | {this_qubit} x {other_qubit}")
                if new_epr:  # swapping succeeded
                    new_epr.src = prev_partner
                    new_epr.dst = next_partner

                    # Keep some info in case of parallel swapping with neighbors:
                    own_rank = swap_sequence[own_idx]
                    prev_p_idx = route.index(prev_partner.name)
                    prev_p_rank = swap_sequence[prev_p_idx]
                    if own_rank == prev_p_rank:
                        # potential parallel swap with prev neighbor
                        self.parallel_swappings[prev_epr.name] = (prev_epr, next_epr, new_epr)

                    next_p_idx = route.index(next_partner.name)
                    next_p_rank = swap_sequence[next_p_idx]
                    if own_rank == next_p_rank:
                        # potential parallel swap with next neighbor
                        self.parallel_swappings[next_epr.name] = (next_epr, prev_epr, new_epr)

                # send SWAP_UPDATE to both swapping partners:
                prev_partner_msg = {
                    "cmd": "SWAP_UPDATE",
                    "path_id": fib_entry["path_id"],
                    "swapping_node": self.own.name,
                    "partner": next_partner.name,
                    "epr": prev_epr.name,
                    "new_epr": new_epr,  # None means swapping failed
                    "fwd": False,
                }
                self.send_msg(dest=prev_partner, msg=prev_partner_msg, route=fib_entry["path_vector"])

                next_partner_msg = {
                    "cmd": "SWAP_UPDATE",
                    "path_id": fib_entry["path_id"],
                    "swapping_node": self.own.name,
                    "partner": prev_partner.name,
                    "epr": next_epr.name,
                    "new_epr": new_epr,  # None means swapping failed
                    # "destination": next_partner.name,
                    "fwd": False,
                }
                self.send_msg(dest=next_partner, msg=next_partner_msg, route=fib_entry["path_vector"])

                # release qubits
                this_qubit.fsm.to_release()
                other_qubit.fsm.to_release()
                ev1 = QubitReleasedEvent(link_layer=self.link_layer, qubit=prev_qubit, t=simulator.tc, by=self)
                ev2 = QubitReleasedEvent(link_layer=self.link_layer, qubit=next_qubit, t=(simulator.tc + 1e-6), by=self)
                simulator.add_event(ev1)
                simulator.add_event(ev2)

        else:  # end-node
            _, qm = self.memory.read(address=qubit.addr, must=True)
            assert isinstance(qm, WernerStateEntanglement)
            assert qm.src is not None
            assert qm.dst is not None
            qubit.fsm.to_release()
            log.debug(f"{self.own}: consume EPR: {qm.name} -> {qm.src.name}-{qm.dst.name} | F={qm.fidelity}")
            self.e2e_count += 1
            self.fidelity += qm.fidelity
            event = QubitReleasedEvent(
                link_layer=self.link_layer, qubit=qubit, e2e=self.own.name == "S", t=simulator.tc, by=self
            )
            simulator.add_event(event)

    def send_msg(self, dest: Node, msg: dict, route: list[str], delay: bool = False):
        own_idx = route.index(self.own.name)
        dest_idx = route.index(dest.name)

        nh = route[own_idx + 1] if dest_idx > own_idx else route[own_idx - 1]
        next_hop = self.own.network.get_node(nh)

        log.debug(f"{self.own.name}: send msg to {dest.name} via {next_hop.name} | msg: {msg}")

        cchannel = self.own.get_cchannel(next_hop)
        classic_packet = ClassicPacket(msg=msg, src=self.own, dest=dest)
        if delay:
            cchannel.send(classic_packet, next_hop=next_hop, delay=cchannel.delay_model.calculate())
        else:
            cchannel.send(classic_packet, next_hop=next_hop)

    def handle_event(self, event: Event) -> None:
        """Handles external simulator events, specifically QubitEntangledEvent instances.
        In ASYNC mode, events are handled immediately. In SYNC mode, events
        are queued if the current phase is EXTERNAL to ensure synchronization is respected.

        Parameters
        ----------
            event (Event): The event to process, expected to be a QubitEntangledEvent.

        """
        from qns.network.protocol.event import QubitEntangledEvent

        if isinstance(event, QubitEntangledEvent):
            if self.own.timing_mode in (TimingModeEnum.ASYNC, TimingModeEnum.LSYNC):
                self.handle_entangled_qubit(event)
            elif self.sync_current_phase == SignalTypeEnum.EXTERNAL:
                # Accept new etg while we are in EXT phase
                # Assume t_coh > t_ext: QubitEntangledEvent events should correspond to different qubits, no redundancy
                self.waiting_qubits.append(event)

    def handle_sync_signal(self, signal_type: SignalTypeEnum):
        """Processes timing signals for SYNC mode. When receiving an INTERNAL phase start signal, all
        previously queued QubitEntangledEvent instances are processed. Updates the current
        synchronization phase to match the received signal type.

        Parameters
        ----------
            signal_type (SignalTypeEnum): The received synchronization signal.

        """
        log.debug(f"{self.own}:[{self.own.timing_mode}] TIMING SIGNAL <{signal_type}>")
        if self.own.timing_mode == TimingModeEnum.SYNC:
            self.sync_current_phase = signal_type
            if signal_type == SignalTypeEnum.INTERNAL:
                # internal phase -> time to handle all entangled qubits
                log.debug(f"{self.own}: there are {len(self.waiting_qubits)} etg qubits to process")
                for event in self.waiting_qubits:
                    self.handle_entangled_qubit(event)
                self.waiting_qubits = []

    def handle_entangled_qubit(self, event):
        """Handles newly entangled qubits based on their path allocation. If the qubit is assigned
        a path (e.g., in buffer-space multiplexing), the method checks its eligibility and,
        if conditions are met, transitions it to the PURIF state and calls `purif` method.
        If the qubit is unassigned (statistical multiplexing), it is currently ignored.

        Parameters
        ----------
            event: Event containing the entangled qubit and its associated metadata (e.g., neighbor).

        """
        if event.qubit.path_id is not None:  # for buffer-space mux
            fib_entry = self.fib.get_entry(event.qubit.path_id)
            if fib_entry:
                if self.eval_qubit_eligibility(fib_entry, event.neighbor.name):
                    self.own.get_qchannel(event.neighbor)  # ensure qchannel exists
                    event.qubit.fsm.to_purif()
                    self.purif(event.qubit, fib_entry, event.neighbor)
            else:
                raise Exception(f"No FIB entry found for path_id {event.qubit.path_id}")
        else:  # for statistical mux
            log.debug("Qubit not allocated to a path. Statistical mux not supported yet.")

    def select_eligible_qubit(self, exc_qchannel: str, path_id: int | None = None) -> MemoryQubit | None:
        """Searches for an eligible qubit in memory that matches the specified path ID and
        is located on a different qchannel than the excluded one. This is used to
        find a swap candidate during entanglement forwarding. Currently returns the first
        matching result found.

        Parameters
        ----------
            exc_qchannel (str): Name of the quantum channel to exclude from the search.
            path_id (int, optional): Identifier for the entanglement path to match.

        Returns
        -------
            Optional[MemoryQubit]: A single eligible memory qubit, if found; otherwise, None.

        """
        qubits = self.memory.search_eligible_qubits(exc_qchannel=exc_qchannel, path_id=path_id)
        if qubits:
            return qubits[0][0]  # pick up one qubit
            # TODO: Other qubit selection
            # (for statistical multiplexing, multipath, quasi-local swapping, etc.)
        return None

    # searches and selects a purif qubit with same path_id and qchannel for purif
    def select_purif_qubit(
        self, exc_address: int, partner: str, qchannel: str, path_id: int, purif_rounds: int
    ) -> MemoryQubit | None:
        """Searches for a candidate qubit in the PURIF state that is ready for purification
        with a given qubit. The candidate must be a different qubit with the same path ID,
        quantum channel, partner, and has undergone the same number of purification rounds
        (i.e., recurrent purification schedule).

        Parameters
        ----------
            exc_address (int): Memory address of the qubit to exclude from the search.
            partner (str): Name of the partner node entangled with the qubit.
            qchannel (str): Name of the quantum channel associated with the qubit.
            path_id (int): Identifier for the entanglement path to match.
            purif_rounds (int): Number of purification rounds completed.

        Returns
        -------
            Optional[MemoryQubit]: A compatible qubit for purification if found, otherwise None.

        """
        qubits = self.memory.search_purif_qubits(
            exc_address=exc_address, partner=partner, qchannel=qchannel, path_id=path_id, purif_rounds=purif_rounds
        )
        if qubits:
            return qubits[0][0]  # pick up one qubit
            # TODO: Other possible qubit selection criteria
        return None

    ###### Helper functions ######
    def get_memory_qubit(self, epr_name: str):
        res = self.memory.get(key=epr_name)
        if res is not None:
            return res[0]
        return None

    def compute_qubit_allocation(self, path: list[str], m_v: list[int], node: str):
        if node not in path:
            return None, None  # Node not in path
        idx = path.index(node)
        left_qubits = m_v[idx - 1] if idx > 0 else None  # Allocate from left channel
        right_qubits = m_v[idx] if idx < len(m_v) else None  # Allocate for right channel
        return left_qubits, right_qubits
