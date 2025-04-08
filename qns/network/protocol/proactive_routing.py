#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Amar Abane
#    National Institute of Standards and Technology, NIST.
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

from typing import Dict, Optional, List
import uuid

from qns.entity.cchannel.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from qns.entity.memory.memory import QuantumMemory
from qns.entity.memory.memory_qubit import MemoryQubit
from qns.entity.node.app import Application
from qns.entity.node.node import Node
from qns.entity.node.qnode import QNode
from qns.entity.node.controller import Controller
from qns.entity.qchannel.qchannel import QuantumChannel, RecvQubitPacket
from qns.models.core.backend import QuantumModel
from qns.network.requests import Request
from qns.simulator.event import Event, func_to_event
from qns.simulator.simulator import Simulator
from qns.network import QuantumNetwork
from qns.models.epr import WernerStateEntanglement
from qns.simulator.ts import Time
import qns.utils.log as log
from qns.network.protocol.fib import ForwardingInformationBase

import copy


class ProactiveRouting(Application):
    """
    ProactiveRouting runs at the network layer of QNodes (routers) and receives instructions from the controller
    It implements the forwarding phase (i.e., entanglement generation and swapping) while the routing is done at the controller. 
    Purification will be in a sepeare process/module.
    """
    def __init__(self):
        super().__init__()
        self.net: QuantumNetwork = None
        self.own: QNode = None
        self.memories: QuantumMemory = None
        
        self.fib: ForwardingInformationBase = ForwardingInformationBase()
        self.link_layer = None

        # so far we can only distinguish between classic and qubit events (not source Entity)
        self.add_handler(self.RecvClassicPacketHandler, [RecvClassicPacket])
        
        self.swap_count = 0
        
        self._qubits_map = {}

    def install(self, node: QNode, simulator: Simulator):
        from qns.network.protocol.link_layer import LinkLayer
        super().install(node, simulator)
        self.own: QNode = self._node
        self.memories: List[QuantumMemory] = self.own.memories
        self.net = self.own.network
        ll_apps = self.own.get_apps(LinkLayer)
        if ll_apps:
            self.link_layer = ll_apps[0]
        else:
            raise Exception("No LinkLayer protocol found")

    def RecvClassicPacketHandler(self, node: Node, event: Event):
        # node is the local node of this app
        if isinstance(event.packet.src, Controller):
            self.handle_control(event)
        elif isinstance(event.packet.src, QNode):
            self.handle_signaling(event)
        else:
            log.warn(f"Unexpected event from entity type: {type(event.packet.src)}")

    # handle forwarding instructions from the controller
    def handle_control(self, packet: RecvClassicPacket):
        log.debug(f"{self.own.name}: received instructions from controller")
        msg = packet.packet.get()
        log.debug(f"msg: {msg}")

        path_id = msg['path_id']
        instructions = msg['instructions']
        # TODO: verify vectors consistency (size, min/max, etc.)

        prev_neighbor = None
        next_neighbor = None
        pn = ""
        nn = ""
        # node gets prev and next node from route vector:
        if self.own.name in instructions['route']:
            i = instructions['route'].index(self.own.name)
            pn, nn = (instructions['route'][i - 1] if i > 0 else None, instructions['route'][i + 1] if i < len(instructions['route']) - 1 else None)
        else:
            raise Exception(f"Node {self.own.name} not found in route vector {instructions['route']}")

        # use prev and next node to get corresponding channels
        # use channel names to get corresponding memories
        prev_qchannel = None
        prev_qmem = None
        if pn:
            prev_neighbor = self.net.get_node(pn)
            prev_qchannel: QuantumChannel = self.own.get_qchannel(prev_neighbor)
            if prev_qchannel:
                prev_qmem = next(qmem for qmem in self.memories if qmem.name == prev_qchannel.name)
            else:
                raise Exception(f"Qchannel not found for neighbor {prev_neighbor}")

        next_qchannel = None
        next_qmem = None
        if nn:
            next_neighbor = self.own.network.get_node(nn)
            next_qchannel: QuantumChannel = self.own.get_qchannel(next_neighbor)
            if next_qchannel:
                next_qmem = next(qmem for qmem in self.memories if qmem.name == next_qchannel.name)
            else:
                raise Exception(f"Qchannel not found for neighbor {next_neighbor}")

        # use mux info to allocate qubits in each memory, keep qubit addresses
        prev_qubits = []
        next_qubits = []
        if instructions['mux'] == "B":
            if instructions["m_v"]:
                print(f"{self.own}: Allocating qubits for buffer-space mux")
                num_prev, num_next = self.compute_qubit_allocation(instructions['route'], instructions['m_v'], self.own.name)
                if num_prev and prev_qmem:
                    if num_prev >= prev_qmem.free:
                        for i in range(num_prev): prev_qubits.append(prev_qmem.allocate(path_id=path_id))
                    else:
                        raise Exception(f"Not enough qubits left for this allocation.")
                if num_next and next_qmem:
                    if num_next >= next_qmem.free:
                        for i in range(num_next): next_qubits.append(next_qmem.allocate(path_id=path_id))
                    else:
                        raise Exception(f"Not enough qubits left for this allocation.")
            else:
                print(f"{self.own}: Qubits allocation not provided. Allocate all qubits")
                if prev_qmem:
                    if prev_qmem.free == prev_qmem.capacity:
                        for i in range(prev_qmem.free): prev_qubits.append(prev_qmem.allocate(path_id=path_id))
                    else:
                        raise Exception(f"Memory {prev_qmem.name} has allocated qubits and cannot be used with Blocking mux.")
                if next_qmem:
                    if next_qmem.free == next_qmem.capacity:
                        for i in range(next_qmem.free): next_qubits.append(next_qmem.allocate(path_id=path_id))
                    else:
                        raise Exception(f"Memory {next_qmem.name} has allocated qubits and cannot be used with Blocking mux.")
        print(f"allocated qubits: prev={prev_qubits} | next={next_qubits}")

        # populate FIB
        if self.fib.get_entry(path_id):
            self.fib.delete_entry(path_id)        
        self.fib.add_entry(path_id=path_id, path_vector=instructions['route'], swap_sequence=instructions['swap'], 
                           purification_scheme=instructions['purif'], qubit_addresses=[])

        # call LINK LAYER to start generating EPRs on next channels: this will trigger "new_epr" events
        if next_neighbor:
            from qns.network.protocol.event import LinkLayerManageActiveChannels, TypeEnum
            t = self._simulator.tc #+ self._simulator.time(sec=0)   # simulate comm. time between L3 and L2
            ll_request = LinkLayerManageActiveChannels(link_layer=self.link_layer, next_hop=next_neighbor, 
                                                       type=TypeEnum.ADD, t=t, by=self)
            self._simulator.add_event(ll_request)
            # log.debug(f"{self.own.name}: calling link layer to generate eprs for path {path_id} with next hop {next_neighbor}")
        
        # TODO: on remove path:
        # update FIB
        # if qchannel is not used by any path -> notify LinkLayer to stop generating EPRs over it:
        #t = self._simulator.tc + self._simulator.time(sec=1e-6)   # simulate comm. time between L3 and L2
        #ll_request = LinkLayerManageActiveChannels(link_layer=self.link_layer, next_hop=next_hop, 
        #                                           type=TypeEnum.REMOVE, t=t, by=self)
        #self._simulator.add_event(ll_request)


    # handle classical message from neighbors
    def handle_signaling(self, packet: RecvClassicPacket):
        msg = packet.packet.get()
        cchannel = packet.cchannel

        from_node: QNode = cchannel.node_list[0] \
            if cchannel.node_list[1] == self.own else cchannel.node_list[1]

        cmd = msg["cmd"]
        path_id = msg["path_id"]

        if cmd == "SWAP_UPDATE":
            fib_entry = self.fib.get_entry(path_id)
            if not fib_entry:
                raise Exception(f"{self.own}: FIB entry not found for path {path_id}")

            if msg["destination"] == self.own.name:    # destination means: the node needs to update its local qubit wrt a remote node (partner)
                if msg["cycle"] == fib_entry["swapped_self"] + 1:     # node did not swap yet for this e2e cycle
                    log.debug(f"{self.own}: rcvd SU cycle={msg['cycle']} and did not swap yet")
                    qmem, qubit = self.get_memory_qubit(msg["epr"])
                    if qmem:
                        qmem.read(address=qubit.addr)       # erase swapped EPR 
                        qmem.write(qm=msg["new_epr"], address=qubit.addr)   # save new EPR with new fidelity and src/dst
                        if self.eval_swapping_conditions(fib_entry, msg["partner"]):
                            log.debug(f"{self.own}: qubit {qubit} go to purif")
                            qubit.fsm.to_purif()
                            self.purif(qmem, qubit, fib_entry, msg["partner"])
                            
                            # TODO[VORA]: push this SU to the other side as new dest if there is a lower, non-zero rank node
                        #else:
                        #    log.debug(f"# {self.own}: is swapping dest and eligibility not met")    
                    else:
                        log.debut(f"# {self.own}: EPR qubit not found")
                elif msg["cycle"] > fib_entry["swapped_self"]:
                    log.debug(f"# {self.own}: desynchronized swapping cycles")
                    return
                else:    # cycle <= swapped_self
                    # node is destination but swapped in the meantime
                    log.debut(f"# {self.own}: is swapping dest but has already swapped")
                    fib_entry = self.fib.get_entry(path_id)
                    msg["results"][self.own] = 1
                    # TODO[PARALLEL]: push to next neighbor as new dest
            else:
                msg_copy = copy.deepcopy(msg)
                # node is not destination of this swap update: forward message
                fib_entry = self.fib.get_entry(path_id)
                if fib_entry["swapped_self"] >= msg["cycle"]:  # if swapped: append result
                    msg_copy["results"][self.own] = 1
                else:
                    log.debut(f"# {self.own}: not the swapping dest but has not swapped yet")

                log.debug(f"{self.own}: FWD SWAP_UPDATE")
                msg_copy["fwd"] = True
                self.send_swap_update(dest=packet.packet.dest, msg=msg_copy, route=fib_entry["path_vector"])

                """ if self.own.name == 'R1':
                    qubits = self._qubits_map[fib_entry["swapped_self"]]
                    prev_qmem, prev_qubit = qubits['prev']
                    next_qmem, next_qubit = qubits['next']

                    from qns.network.protocol.event import QubitReleasedEvent
                    light_speed = 2 * 10**5 # km/s
                    prev_t = self._simulator.tc #+ self._simulator.time(sec = 1e-6)
                    next_t = self._simulator.tc + self._simulator.time(10 / light_speed)
                    prev_ev = QubitReleasedEvent(link_layer=self.link_layer, qubit=prev_qubit, t=prev_t, by=prev_qmem)
                    next_ev = QubitReleasedEvent(link_layer=self.link_layer, qubit=next_qubit, t=next_t, by=next_qmem)
                    self._simulator.add_event(prev_ev)
                    self._simulator.add_event(next_ev) """


    # handle internal events
    def handle_event(self, event: Event) -> None:
        from qns.network.protocol.event import QubitEntangledEvent
        if isinstance(event, QubitEntangledEvent):    # this event starts the lifecycle for a qubit
            if event.qubit.pid is not None:     # expected with buffer-space / blocking
                fib_entry = self.fib.get_entry(event.qubit.pid)
                if fib_entry:
                    if self.eval_swapping_conditions(fib_entry, event.neighbor.name):
                        qchannel: QuantumChannel = self.own.get_qchannel(event.neighbor)
                        if qchannel:
                            # log.debug(f"{self.own}: Move qubit {event.qubit} to PURIF")
                            event.qubit.fsm.to_purif()
                            qmem = next(qmem for qmem in self.memories if qmem.name == qchannel.name)
                            self.purif(qmem, event.qubit, fib_entry, event.neighbor.name)
                        else:
                            raise Exception(f"No qchannel found for neighbor {event.neighbor.name}")
                else:
                    raise Exception(f"No FIB entry found for pid {event.qubit.pid}")
            else:
                log.debug("Qubit not allocated to any path. Statistical mux not supported yet.")     

    
    # eval qubit eligibility 
    def eval_swapping_conditions(self, fib_entry: Dict, partner: str) -> bool:
        route = fib_entry['path_vector']
        swap_sequence = fib_entry['swap_sequence']
        partner_idx = route.index(partner)
        partner_rank = swap_sequence[partner_idx]
        own_idx = route.index(self.own.name)
        own_rank = swap_sequence[own_idx]

        # If partner rank is higher or equal -> go to PURIF
        if partner_rank >= own_rank:
            return True
        return False

    def purif(self, qmem: QuantumMemory, qubit: MemoryQubit, fib_entry: Dict, partner: str):
        # Will remove when purif cycle is implemented:
        # log.debug(f"Skip purif -> OK")
        qubit.fsm.to_eligible()
        self.eligible(qmem, qubit, fib_entry)

        # TODO:
        # get partner's rank: if strictly higher -> i am init purif
        # To init purif: 
        #   - get purif_scheme for the segment (own-partner)
        #   - if num_rounds (or condition) is None/satified: return ELIGIBLE for this qubit
        #   - else: check if available EPRs (same path_id and same partner)
        #           if enough pairs -> enter purif cycle (passive slightly behind active in transitions)
        #           else -> return

    def eligible(self, qmem: QuantumMemory, qubit: MemoryQubit, fib_entry: Dict):
        route = fib_entry['path_vector']
        own_idx = route.index(self.own.name)
        if own_idx > 0 and own_idx < len(route)-1:     # intermediate node
            (other_qmem, qubits) = self.check_eligible_qubit(qmem, fib_entry['path_id'])   # check if there is another eligible qubit
            if not other_qmem:
                # print(f"{self.own}: no eligible qubit for now")
                return
            else:     # do swapping
                other_qubit, other_epr = qubits[0]     # pick up one qubit
                this_epr = qmem.get(address=qubit.addr)[1]

                log.debug(f"{self.own}: SWAP {qmem}.{qubit} x {other_qmem}.{other_qubit}")
                new_epr = this_epr.swapping(epr=other_epr, name=uuid.uuid4().hex)

                # increment cycle
                self.fib.update_entry(fib_entry["path_id"], swapped_self=fib_entry["swapped_self"]+1)
                self.fib.update_received_swaps(fib_entry["path_id"], self.own.name, 1)

                qmem.read(address=qubit.addr)
                other_qmem.read(key=other_qubit.addr)
                qubit.fsm.to_release()
                other_qubit.fsm.to_release()

                if new_epr:    # order eprs and prev/next nodes
                    prev_qubit = None
                    next_qubit = None
                    prev_qmem = None
                    next_qmem = None
                    if this_epr.dst == self.own:
                        prev_partner = this_epr.src
                        prev_epr = this_epr
                        next_partner = other_epr.dst
                        next_epr = other_epr
                        
                        prev_qubit = qubit
                        next_qubit = other_qubit
                        prev_qmem = qmem
                        next_qmem = other_qmem
                    elif this_epr.src == self.own:
                        prev_partner = other_epr.src
                        prev_epr = other_epr
                        next_partner = this_epr.dst
                        next_epr = this_epr
                        
                        prev_qubit = other_qubit
                        next_qubit = qubit
                        prev_qmem = other_qmem
                        next_qmem = qmem
                    else:
                        raise Exception(f"Unexpected: swapping EPRs {this_epr} x {other_epr}")
                    
                    #if self.own.name == 'R2':
                    from qns.network.protocol.event import QubitReleasedEvent
                    #light_speed = 2 * 10**5 # km/s
                    prev_t = self._simulator.tc # + self._simulator.time(sec = 1e-6)
                    next_t = self._simulator.tc # + self._simulator.time(sec = 1e-6) # + 30 / light_speed)
                    prev_ev = QubitReleasedEvent(link_layer=self.link_layer, qubit=prev_qubit, t=prev_t, by=prev_qmem)
                    next_ev = QubitReleasedEvent(link_layer=self.link_layer, qubit=next_qubit, t=next_t, by=next_qmem)
                    self._simulator.add_event(prev_ev)
                    self._simulator.add_event(next_ev)
                    #else:
                    #    self._qubits_map[fib_entry["swapped_self"]] = { 'prev': (prev_qmem, prev_qubit), 'next': (next_qmem, next_qubit) }

                    new_epr.src = prev_partner
                    new_epr.dst = next_partner

                    # send SWAP_UPDATE to both swapping partners: 
                    prev_partner_msg = {
                        "cmd": "SWAP_UPDATE",
                        "path_id": fib_entry["path_id"],
                        "swapping_node": self.own.name,
                        "partner": next_partner.name,
                        "epr": prev_epr.name,
                        "new_epr": new_epr,
                        "results": fib_entry["received_swaps"],
                        "cycle": fib_entry["swapped_self"],
                        "destination": prev_partner.name
                    }
                    prev_partner_msg["fwd"] = False
                    self.send_swap_update(dest=prev_partner, msg=prev_partner_msg, route=fib_entry["path_vector"])
                
                    next_partner_msg = {
                        "cmd": "SWAP_UPDATE",
                        "path_id": fib_entry["path_id"],
                        "swapping_node": self.own.name,
                        "partner": prev_partner.name,
                        "epr": next_epr.name,
                        "new_epr": new_epr,
                        "results": fib_entry["received_swaps"],
                        "cycle": fib_entry["swapped_self"],
                        "destination": next_partner.name
                    }
                    next_partner_msg["fwd"] = False
                    self.send_swap_update(dest=next_partner, msg=next_partner_msg, route=fib_entry["path_vector"])
                else:
                    print("send SWAP_UPDATE failed")
        else: # end-node
            from qns.network.protocol.event import QubitReleasedEvent
            # log.debug(f"{self.own}: Deliver qubit to app.")
            qmem.read(address=qubit.addr)
            qubit.fsm.to_release()
            self.fib.update_entry(fib_entry["path_id"], swapped_self=fib_entry["swapped_self"]+1)
            t = self._simulator.tc #+ self._simulator.time(sec=0)
            event = QubitReleasedEvent(link_layer=self.link_layer, qubit=qubit, t=t, by=qmem)
            self._simulator.add_event(event)


    def send_swap_update(self, dest: Node, msg: Dict, route: List[str], delay: float = 0):
        log.debug(f"{self.own.name}: send_SU {dest} = {msg}")
        own_idx = route.index(self.own.name)        
        dest_idx = route.index(dest.name)

        nh = route[own_idx+1] if dest_idx > own_idx else route[own_idx-1]
        next_hop = self.own.network.get_node(nh)
        
        cchannel: ClassicChannel = self.own.get_cchannel(next_hop)
        if cchannel is None:
            raise Exception(f"{self.own}: No classic channel for dest {dest}")

        classic_packet = ClassicPacket(msg=msg, src=self.own, dest=dest)
        cchannel.send(classic_packet, next_hop=next_hop, delay=delay)
        # log.debug(f"{self.own}: send SWAP_UPDATE to {dest} via {next_hop}")


    def check_eligible_qubit(self, qmem: QuantumMemory, path_id: int = None):
        # assume isolated paths -> a path_id uses only left and right qmem
        for qm in self.memories:
            if qm.name != qmem.name:
                qubits = qm.search_eligible_qubits(pid=path_id)
                if qubits:
                    return qm, qubits
        return None, None
    
    def get_memory_qubit(self, epr_name: str):
        for qm in self.memories:
            res = qm.get(key=epr_name)
            if res is not None:
                return qm, res[0]
        return None, None

    def compute_qubit_allocation(self, path, m_v, node):
        if node not in path:
            return None, None           # Node not in path
        idx = path.index(node)
        prev_qubits = m_v[idx - 1] if idx > 0 else None  # Allocate from previous channel
        next_qubits = m_v[idx] if idx < len(m_v) else None  # Allocate for next channel
        return prev_qubits, next_qubits