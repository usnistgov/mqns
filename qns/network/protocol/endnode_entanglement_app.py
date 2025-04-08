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

from typing import Dict, Optional
import uuid

from qns.entity.cchannel.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from qns.entity.memory.memory import QuantumMemory
from qns.entity.node.app import Application
from qns.entity.node.node import QNode
from qns.entity.qchannel.qchannel import QuantumChannel, RecvQubitPacket
from qns.models.core.backend import QuantumModel
from qns.network.requests import Request
from qns.simulator.event import Event, func_to_event
from qns.simulator.simulator import Simulator
from qns.network import QuantumNetwork
from qns.models.epr import WernerStateEntanglement
from qns.simulator.ts import Time
import qns.utils.log as log


class Transmit():
    def __init__(self, id: str, src: QNode, dst: QNode,
                 first_epr_name: Optional[str] = None, second_epr_name: Optional[str] = None):
        self.id = id
        self.src = src
        self.dst = dst
        self.first_epr_name = first_epr_name
        self.second_epr_name = second_epr_name

    def __repr__(self) -> str:
        return f"<transmit {self.id}: {self.src} -> {self.dst},\
             epr: {self.first_epr_name}, {self.second_epr_name}>"


class EndNodeEntanglementApp(Application):     # application to request entanglements then withdraw
    def __init__(self, name: str, src, dest, attr: Dict = {}):
        super().__init__()
        self.name = name
        self.own: QNode = None
        self.src: QNode = src
        self.dest: QNode = dest
        self.attr: Dict = attr
        
        self.net: QuantumNetwork = None

        self.requests = []

        self.success = []
        self.success_count = 0
        self.send_count = 0

        self.add_handler(self.RecvClassicPacketHandler, [RecvClassicPacket], [Controller])
        self.add_handler(self.RecvClassicPacketHandler, [RecvClassicPacket], [ProactiveRoutingControllerApp])

    def install(self, node: QNode, simulator: Simulator):
        super().install(node, simulator)
        self.own: QNode = self._node
        self.controller = self.own.network.controller

        if self.dst is not None:
            # I am a sender
            t = simulator.ts
            event = func_to_event(t, self.submit_request, by=self)
            self._simulator.add_event(event)

    def RecvClassicPacketHandler(self, node: QNode, event: Event):
        self.handle_reponse(event)
    
    def RecvNewPair(self, app: ProactiveRouting, event: Event):
        self.handle_pair(event)

    def submit_request(self):      # send request to the controller
        # insert the cancel request event
        t = self._simulator.tc + Time(sec=1 / self.send_rate)
        event = func_to_event(t, self.withdraw_request, by=self)
        self._simulator.add_event(event)

        log.debug(f"{self.own}: submit new request")
        self.send_count += 1
        
        # TODO: create and save request
        
        # get channel to the controller
        cchannel: ClassicChannel = self.own.get_cchannel(self.controller)
        if cchannel is None:
            raise Exception("No such classic channel to the controller")
        
        classic_packet = ClassicPacket(
            msg={"cmd": "submit_request", "attrs": {}}, src=self.own, dest=self.controller)
        cchannel.send(classic_packet, next_hop=self.controller)
        log.debug(f"{self.own}: send {classic_packet.msg} from {self.own} to {self.controller}")

    def withdraw_request(self):
        log.debug(f"{self.own}: withdraw request")
        
        # TODO: find the request
        
        # get channel to the controller
        cchannel: ClassicChannel = self.own.get_cchannel(self.controller)
        if cchannel is None:
            raise Exception("No such classic channel to the controller")
        
        classic_packet = ClassicPacket(
            msg={"cmd": "withdraw_request", "id": 0}, src=self.own, dest=self.controller)
        cchannel.send(classic_packet, next_hop=self.controller)
        log.debug(f"{self.own}: send {classic_packet.msg} from {self.own} to {self.controller}")

    def handle_reponse(self, packet: RecvClassicPacket):      # handle response from controller
        msg = packet.packet.get()
        cchannel = packet.cchannel

        from_node: QNode = cchannel.node_list[0] \
            if cchannel.node_list[1] == self.own else cchannel.node_list[1]

        log.debug(f"{self.own}: recv {msg} from {from_node}")

        cmd = msg["cmd"]

        if cmd == "new_pair":
            success_count+=1
            
            
    def handle_pair(self, packet: RecvClassicPacket):      # handle pairs from routing protocol
        msg = packet.packet.get()
        cchannel = packet.cchannel

        from_node: QNode = cchannel.node_list[0] \
            if cchannel.node_list[1] == self.own else cchannel.node_list[1]

        log.debug(f"{self.own}: recv {msg} from {from_node}")

        cmd = msg["cmd"]

        if cmd == "new_pair":
            success_count+=1


    # see what to do with this:
    def add_request(self, request):
        """
        add a request to this app

        Args:
            request (Request): the inserting request
        """
        self.requests.append(request)

    def clear_request(self):
        """
        clear all requests
        """
        self.requests.clear()
