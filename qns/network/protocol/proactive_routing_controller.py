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

from qns.entity.node.controller import Controller
from qns.entity.node.qnode import QNode

from qns.entity.cchannel.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from qns.entity.memory.memory import QuantumMemory
from qns.entity.node.app import Application

from qns.entity.qchannel.qchannel import QuantumChannel, RecvQubitPacket
from qns.models.core.backend import QuantumModel
from qns.network.requests import Request
from qns.simulator.event import Event, func_to_event
from qns.simulator.simulator import Simulator
from qns.network import QuantumNetwork
from qns.simulator.ts import Time
import qns.utils.log as log

from http.server import BaseHTTPRequestHandler, HTTPServer


class ProactiveRoutingControllerApp(Application):
    def __init__(self):
        super().__init__()
        self.net: QuantumNetwork = None           # contains QN physical topology and classical topology 
        self.own: Controller = None               # controller node

        self.add_handler(self.RecvClassicPacketHandler, [RecvClassicPacket])       # E2E etg. requests sent from end-nodes to the controller 

        self.server = HTTPServer(('', 8080), self.RequestHandler)
        self.RequestHandler.test = self.test  # Pass test method to handler

    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.test()  # Call the test method of MyServer instance
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Test function executed")
    
    def install(self, node: Controller, simulator: Simulator):
        super().install(node, simulator)
        self.own: Controller = self._node
        self.net = self.own.network
        
        #print("Starting server on port 8080...")
        #self.server.serve_forever()

        # send a test control to QNodes
        t = self._simulator.tc # + self._simulator.time(sec=1)
        event = func_to_event(t, self.test, by=self)
        self._simulator.add_event(event)

    def RecvQubitHandler(self, node: QNode, event: Event):
        self.response_distribution(event)

    def test(self):
        self.net.build_route()
        network_nodes = self.net.get_nodes()
        
        src = None
        dst = None
        for qn in network_nodes:
            if qn.name == "S":
                src = qn
            if qn.name == "D":
                dst = qn
        
        route_result = self.net.query_route(src, dst)
        path_nodes = route_result[0][2]
        print(f"{self.own}: Computed path: {path_nodes}")
        
        route = [n.name for n in path_nodes]
        
        # for buffer-space mux -> get memory capacities per channel
        m_v = []
        for i in range(len(path_nodes) - 1):
            channel_name = f"q_{path_nodes[i].name},{path_nodes[i+1].name}"
            qmemory = next(qmem for qmem in self.net.get_node(path_nodes[i].name).memories if qmem.name == channel_name)
            num_qubits = qmemory.capacity
            m_v.append(num_qubits)

        for qnode in path_nodes:
            instructions = {
                "route": route,
               # "swap": [1,0,1],
                "swap": [2,0,1,2],
                "mux": "B",      # or "S" for statistical
                "m_v": m_v,     # Optional: to represent buffer-sapce
                "purif": { "S,R": 1, "R,D": 1, "S,D": 1 }    # e.g., purif at all nesting levels
            }

            cchannel = self.own.get_cchannel(qnode)
            classic_packet = ClassicPacket(
                msg={"cmd": "install_path", "path_id": 0, "instructions": instructions}, src=self.own, dest=qnode)
            cchannel.send(classic_packet, next_hop=qnode)
            log.debug(f"{self.own}: send {classic_packet.msg} to {qnode}")


    def RecvClassicPacketHandler(self, node: Controller, event: Event):
        self.handle_request(event)

    def handle_request(self, event: RecvClassicPacket):
        msg = event.packet.get()
        cchannel = event.cchannel

        from_node: Node = cchannel.node_list[0] \
            if cchannel.node_list[1] == self.own else cchannel.node_list[1]

        log.debug(f"{self.own}: recv {msg} from {from_node}")

        cmd = msg["cmd"]
        request_id = msg["request_id"]

        if cmd == "submit":
            # process new request submitted and send instructions to QNodes
            # can model processing time with events
            nodes_in_path = []
            for qnode in nodes_in_path:
                classic_packet = ClassicPacket(
                    msg={"cmd": "install_path", "request_id": request_id}, src=self.own, dest=qnode)
                cchannel.send(classic_packet, next_hop=qnode)
                log.debug(f"{self.own}: send {classic_packet.msg} to {qnode}")
        elif cmd == "withdraw":
            # remove request and send instructions to QNodes
            pass
        else:
            pass

