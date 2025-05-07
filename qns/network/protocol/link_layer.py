#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2024-2025 Amar Abane
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

from typing import Dict, Optional
import uuid

from qns.entity.memory.memory_qubit import MemoryQubit
from qns.entity.cchannel.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from qns.entity.memory.memory import QuantumMemory
from qns.entity.node.app import Application
from qns.entity.node.node import Node
from qns.entity.node.qnode import QNode
from qns.entity.qchannel.qchannel import QuantumChannel, RecvQubitPacket
from qns.models.core.backend import QuantumModel
from qns.simulator.event import Event, func_to_event
from qns.simulator.simulator import Simulator
from qns.network import QuantumNetwork, TimingModeEnum, SignalTypeEnum
from qns.models.epr import WernerStateEntanglement
from qns.simulator.ts import Time
import qns.utils.log as log

import numpy as np

light_speed = 2 * 10**5 # km/s

def simulate_total_time_until_success(p, round_duration):
    total_time = 0
    total_rounds = 0
    success = False
    while not success:
        if np.random.rand() < np.sqrt(p):  # first round success
            total_rounds+=1
            total_time += round_duration
            if np.random.rand() < np.sqrt(p):  # second round success
                total_rounds+=1
                total_time += round_duration
                success = True
            else:
                total_rounds+=1
                total_time += round_duration
        else:
            total_rounds+=1
            total_time += round_duration
    return total_time, total_rounds


class LinkLayer(Application):
    """
    LinkLayer runs at the link layer of QNodes (routers) and receives instructions from the network layer.
    It implements the EPR generation over individual qchannels.
    """
    def __init__(self, 
                 attempt_rate: int = 1e6,         # attempt_rate ~ min(fiber frequency, count rate)
                 alpha_db_per_km: float = 0.2, 
                 eta_d: float = 1.0, eta_s: float = 1.0, 
                 frequency: int = 80e6,
                 init_fidelity: int = 0.99):
        super().__init__()
        self.sync_current_phase = SignalTypeEnum.EXTERNAL

        self.alpha_db_per_km = alpha_db_per_km
        self.eta_s = eta_s
        self.eta_d = eta_d
        self.frequency = frequency
        self.init_fidelity = init_fidelity
        self.attempt_rate = attempt_rate     # ~ source rate (i.e., qchannel)

        self.own: QNode = None
        self.memory: QuantumMemory = None
        self.net_layer = None
        
        self.active_channels = {}       # stores the qchannels that are part of an installed path
        self.waiting_channels = {}      # stores the qchannels that have all their qubits waiting for the next EXTERNAL phase (LSYNC mode)
        self.waiting_qubits = set()        # stores the qubits waiting for the next EXTERNAL phase (LSYNC mode)

        # so far we can only distinguish between classic and qubit events (not source Entity)
        self.add_handler(self.RecvQubitHandler, [RecvQubitPacket])
        self.add_handler(self.RecvClassicPacketHandler, [RecvClassicPacket])
        
        self.pending_negoc = {}
        self.fifo_epr_init = []
        
        self.etg_count = 0
        self.decoh_count = 0

    def install(self, node: QNode, simulator: Simulator):
        from qns.network.protocol.proactive_routing import ProactiveRouting
        super().install(node, simulator)
        self.own: QNode = self._node
        self.memory: QuantumMemory = self.own.memory
        nl_apps = self.own.get_apps(ProactiveRouting)
        if nl_apps:
            self.net_layer = nl_apps[0]
        else:
            raise Exception("No NetworkLayer protocol found")

    def RecvQubitHandler(self, node: QNode, event: Event):
        self.handle_distribution(event)

    def RecvClassicPacketHandler(self, node: Node, event: Event):
        if event.packet.get()["cmd"] in ["epr_succeeded", "epr_failed"]:
            self.handle_signaling(event)
        elif event.packet.get()["cmd"] in ["epr_init", "epr_ok", "epr_nok"]:
            self.handle_negociation(event)

    def handle_active_channel(self, qchannel: QuantumChannel, next_hop: QNode):
        # use qchannel name to get memory
        qubits = self.memory.get_channel_qubits(qchannel.name)
        log.debug(f"{self.own}: {qchannel.name} has assigned qubits: {qubits}")
        for i, (qb, data) in enumerate(qubits):
            if data is None:
                t = self._simulator.tc + Time(sec = i * 1 / self.attempt_rate)
                event = func_to_event(t, self.start_negociation, by=self, 
                                      next_hop=next_hop, qchannel=qchannel,
                                      qubit=qb, path_id=qb.pid)
                self._simulator.add_event(event)
            else:
                raise Exception(f"{self.own}: --> PROBLEM {data}")

    def start_negociation(self, next_hop: Node, qchannel: QuantumChannel,
                          qubit: MemoryQubit, path_id: Optional[int] = None):
        key = self.own.name +'_'+ next_hop.name
        if path_id is not None:
            key = key+'_'+str(path_id)
        key = key+'_'+str(qubit.addr)

        if key in self.pending_negoc:
            raise Exception(f"{self.own}: negociation already init for {key}")
            return

        log.debug(f"{self.own}: init negociation for {key}")
        qubit.active = key
        self.pending_negoc[key] = (qchannel, next_hop, qubit.addr)
        cchannel: ClassicChannel = self.own.get_cchannel(next_hop)
        if cchannel is None:
            raise Exception(f"{self.own}: No classic channel for dest {dest}")
        classic_packet = ClassicPacket(msg={"cmd": "epr_init", "path_id": path_id, "key": key}, src=self.own, dest=next_hop)
        cchannel.send(classic_packet, next_hop=next_hop)

    def generate_entanglement(self, qchannel: QuantumChannel, next_hop: Node, 
                              address: int, key: str):
        if qchannel.name not in self.active_channels:
            raise Exception(f"{self.own}: Qchannel not active")
            return

        t_mem = 1 / self.memory.decoherence_rate
        if qchannel.length >= (2 * light_speed * t_mem):
            raise Exception("Qchannel too long for entanglement attempt.")

        succ_attempt_time, attempts = self.skip_ahead_entanglement(qchannel.length)
        t_event = self._simulator.tc + Time(sec = succ_attempt_time)
        event = func_to_event(t_event, self.do_successful_attempt, by=self, qchannel=qchannel, 
                              next_hop=next_hop, address=address, attempts=attempts, key=key)
        self._simulator.add_event(event)

    def do_successful_attempt(self, qchannel: QuantumChannel, next_hop: Node, 
                              address, attempts: int, key: str):
        epr = self.generate_epr(next_hop)
        epr.attempts = attempts
        epr.key = key

        # if 3-4 tau -> we are at 3tau
        #local_qubit = qmemory.write(qm=epr, address=address, delay=qchannel.delay_model.calculate())   # qubit init at 2tau
        
        # if 3-6 tau -> we are at 5tau
        local_qubit = self.memory.write(qm=epr, address=address, delay=3*qchannel.delay_model.calculate())   # qubit init at 2tau

        if not local_qubit:
            raise Exception(f"{self.own}: (sender) Do succ EPR -> memory full, key ({key})")

        epr.path_id = local_qubit.pid
        qchannel.send(epr, next_hop)    # no drop (deterministic)
        self.etg_count+=1
        self.notify_entangled_qubit(neighbor=next_hop, qubit=local_qubit, delay=qchannel.delay_model.calculate())   # wait 1tau to notify

    # handle half-EPR arriving from a neighbor
    def handle_distribution(self, packet: RecvQubitPacket):
        if self.own.timing_mode == TimingModeEnum.SYNC and self.sync_current_phase != SignalTypeEnum.EXTERNAL:
            log.debug(f"{self.own}: EXT phase is over -> stop attempts")
            return

        qchannel: QuantumChannel = packet.qchannel
        from_node: Node = qchannel.node_list[0] \
            if qchannel.node_list[1] == self.own else qchannel.node_list[1]

        cchannel: ClassicChannel = self.own.get_cchannel(from_node)
        if cchannel is None:
            raise Exception("No such classic channel")

        epr: WernerStateEntanglement = packet.qubit

        log.debug(f"{self.own}: recv half-EPR {epr.name} from {from_node} | reservation key {epr.key}")

        # if 3-4 tau -> we are at 4tau
        # local_qubit = qmemory.write(qm=epr, pid=epr.path_id, delay=2*qchannel.delay_model.calculate())   # qubit init at 2tau

        # if 3-6 tau -> we are at 6tau
        local_qubit = self.memory.write(qm=epr, pid=epr.path_id, key=epr.key, delay=4*qchannel.delay_model.calculate())   # qubit init at 2tau

        if local_qubit is None:
            raise Exception(f"{self.own}: Failed to store rcvd EPR due to full memory")

        self.notify_entangled_qubit(neighbor=from_node, qubit=local_qubit)

    # handle classical message from neighbors
    def handle_signaling(self, packet: RecvClassicPacket):
        if self.own.timing_mode == TimingModeEnum.SYNC and self.sync_current_phase != SignalTypeEnum.EXTERNAL:
            log.debug(f"{self.own}: EXT phase is over -> stop attempts")
            return

        msg = packet.packet.get()
        cchannel = packet.cchannel
        from_node: QNode = cchannel.node_list[0] \
            if cchannel.node_list[1] == self.own else cchannel.node_list[1]

        cmd = msg["cmd"]
        path_id = msg["path_id"]
        epr_id = msg["epr_id"]
        
        qchannel: QuantumChannel = self.own.get_qchannel(from_node)
        if qchannel is None:
            raise Exception("No such quantum channel")

        # ignore if qchannel not active anymore
        if qchannel.name not in self.active_channels:
            log.debug(f"{self.own}: Qchannel not active anymore")
            (qubit, _) = qmemory.read(epr_id)    # this will free up the qubit of this epr
            return

    def generate_epr(self, dst: QNode) -> QuantumModel:
        epr = WernerStateEntanglement(fidelity=self.init_fidelity, name=uuid.uuid4().hex)
        epr.src = self.own
        epr.dst = dst
        return epr

    def notify_entangled_qubit(self, neighbor: QNode, qubit: MemoryQubit, delay: float = 0):
        from qns.network.protocol.event import QubitEntangledEvent
        qubit.fsm.to_entangled()
        t = self._simulator.tc + self._simulator.time(sec=delay)
        event = QubitEntangledEvent(net_layer=self.net_layer, neighbor=neighbor, qubit=qubit, t=t, by=self)
        self._simulator.add_event(event)
    
    # handle internal events
    def handle_event(self, event: Event) -> None:
        from qns.network.protocol.event import LinkLayerManageActiveChannels, TypeEnum, \
            QubitDecoheredEvent, QubitReleasedEvent
        if isinstance(event, LinkLayerManageActiveChannels):
            log.debug(f"{self.own}: start qchannel with {event.next_hop}")
            qchannel: QuantumChannel = self.own.get_qchannel(event.next_hop)
            if qchannel is None:
                raise Exception("No such quantum channel")
            if event.type == TypeEnum.ADD:
                if qchannel.name not in self.active_channels:
                    self.active_channels[qchannel.name] = (qchannel, event.next_hop)
                    if self.own.timing_mode == TimingModeEnum.ASYNC:
                        self.handle_active_channel(qchannel, event.next_hop)
                    elif self.own.timing_mode == TimingModeEnum.LSYNC:     # LSYNC
                        self.waiting_channels[qchannel.name] = (qchannel, event.next_hop)
                else:
                    raise Exception("Qchannel already handled")
            else:
                self.active_channels.pop(qchannel.name, 'Not Found')
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
                    self.start_negociation(next_hop=next_hop, qchannel=qchannel,
                                           qubit=event.qubit, path_id=event.qubit.pid)
                else:
                    event.qubit.active = None
                    self.check_pending_epr_init()
            else:
                raise Exception("TODO")
        elif isinstance(event, QubitReleasedEvent):
            # check if this node is the EPR initiator of the qchannel associated with the memory of this qubit
            if event.qubit.qchannel.name:
                if event.qubit.qchannel.name in self.active_channels:     # i.e., this node is primary
                    qchannel, next_hop = self.active_channels[event.qubit.qchannel.name]
                    if self.own.timing_mode == TimingModeEnum.ASYNC:
                        self.start_negociation(next_hop=next_hop, qchannel=qchannel,
                                               qubit=event.qubit, path_id=event.qubit.pid)
                    elif self.own.timing_mode == TimingModeEnum.LSYNC:    # LSYNC
                        entry = (qchannel, next_hop, event.qubit.qchannel.name, event.qubit.addr)
                        self.waiting_qubits.add(entry)
                else:
                    event.qubit.active = None
                    self.check_pending_epr_init()
            else:
                raise Exception("TODO")

    def handle_sync_signal(self, signal_type: SignalTypeEnum):
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

    def handle_negociation(self, packet: RecvClassicPacket):
        msg = packet.packet.get()
        cchannel = packet.cchannel
        from_node: QNode = cchannel.node_list[0] if cchannel.node_list[1] == self.own else cchannel.node_list[1]
        qchannel: QuantumChannel = self.own.get_qchannel(from_node)
        if qchannel is None:
            raise Exception("No such quantum channel")

        cmd = msg["cmd"]
        path_id = msg["path_id"]
        key = msg["key"]
        if cmd == 'epr_init':
            log.debug(f"{self.own}: rcvd epr_init {key}")
            avail_qubits = self.memory.search_path_qubits(path_id)
            if avail_qubits:
                log.debug(f"{self.own}: direct found available qubit for {key}")
                avail_qubits[0].active = key
                classic_packet = ClassicPacket(msg={"cmd": 'epr_ok', "path_id": path_id, "key": key}, 
                                               src=self.own, dest=from_node)
                cchannel.send(classic_packet, next_hop=from_node)
            else:
                log.debug(f"{self.own}: didn't find available qubit for {key}")
                self.fifo_epr_init.append((key, path_id, cchannel, from_node))
        elif cmd == 'epr_ok':
            log.debug(f"{self.own}: returned qubit available with {key}")
            (qchannel, next_hop, address) = self.pending_negoc[key]
            self.generate_entanglement(qchannel=qchannel, next_hop=next_hop, address=address, key=key)
            self.pending_negoc.pop(key, None)
        
    def check_pending_epr_init(self):
        if self.fifo_epr_init:
            (key, path_id, cchannel, from_node) = self.fifo_epr_init[0]
            log.debug(f"{self.own}: handle pending negoc {key}")
            avail_qubits = self.memory.search_path_qubits(path_id)
            if avail_qubits:
                log.debug(f"{self.own}: found available qubit for {key}")
                avail_qubits[0].active = key
                classic_packet = ClassicPacket(msg={"cmd": 'epr_ok', "path_id": path_id, "key": key}, 
                                               src=self.own, dest=from_node)
                cchannel.send(classic_packet, next_hop=from_node)
                self.fifo_epr_init.pop(0)
                
                
    def loss_based_success_prob(self, link_length_km):
        """Compute success probability from fiber loss model for heralded entanglement."""
        p_bsa = 0.5
        p_fiber = 10 ** (-self.alpha_db_per_km * link_length_km / 10)
        p = p_bsa * (self.eta_s**2) * (self.eta_d**2) * p_fiber
        return p

    def skip_ahead_entanglement(self, link_length_km: float):
        reset_time = 1 / self.frequency
        tau = link_length_km / light_speed

        # probability assumes that each attempt has always 2-rounds
        p = self.loss_based_success_prob(link_length_km)
        k = np.random.geometric(p)     # k-th attempt will succeed

        attempt_duration = max(4.5*tau, reset_time)

        # calculate time right before the successful trial
        # the last 1-tau of the successful trial will be executed
        # t_success = ((k-1) * attempt_duration) + 3*tau      # if 3-4 tau
        t_success = ((k-1) * attempt_duration) + (5*tau) # - 2*tau    # if 3-6 tau (can subsctract 2tau from Reservation)
        return t_success, k