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

from qns.entity.cchannel import ClassicPacket
from qns.entity.node import Application, Controller, Node
from qns.network import QuantumNetwork
from qns.simulator import Simulator
from qns.utils import log

# from http.server import BaseHTTPRequestHandler, HTTPServer

swapping_settings = {
    # disable swapping (for studying isolated links)
    "isolation_1": [0, 0, 0],
    # for 1-repeater
    "swap_1": [1, 0, 1],
    # for 2-repeater
    "swap_2_asap": [1, 0, 0, 1],
    "swap_2_l2r": [2, 0, 1, 2],
    "swap_2_r2l": [2, 1, 0, 2],
    # for 3-repeater
    "swap_3_asap": [1, 0, 0, 0, 1],
    "swap_3_baln": [2, 0, 1, 0, 2],
    "swap_3_l2r": [3, 0, 1, 2, 3],
    "swap_3_r2l": [3, 2, 1, 0, 3],
    "swap_3_vora_uniform": [3, 0, 2, 1, 3],  # equiv. [2,0,1,0,2] ~ baln
    "swap_3_vora_increasing": [3, 0, 1, 2, 3],
    "swap_3_vora_decreasing": [3, 2, 1, 0, 3],
    "swap_3_vora_mid_bottleneck": [3, 1, 2, 0, 3],  # [2,0,1,0,2]  ~ baln
    # for 4-repeater
    "swap_4_asap": [1, 0, 0, 0, 0, 1],
    "swap_4_baln": [3, 0, 1, 0, 2, 3],
    "swap_4_l2r": [4, 0, 1, 2, 3, 4],
    "swap_4_r2l": [4, 3, 2, 1, 0, 4],
    "swap_4_vora_uniform": [4, 0, 3, 1, 2, 4],  # equiv. [3,0,2,0,1,3]
    "swap_4_vora_increasing": [4, 0, 1, 3, 2, 4],  # equiv. [3,0,1,2,0,3]
    "swap_4_vora_decreasing": [4, 3, 1, 2, 0, 4],  # equiv. [3,2,0,1,0,3]
    "swap_4_vora_mid_bottleneck": [4, 0, 2, 3, 1, 4],  # equiv. [3,0,1,2,0,3]
    "swap_4_vora_uniform2": [3, 0, 2, 0, 1, 3],
    "swap_4_vora_increasing2": [3, 0, 1, 2, 0, 3],
    "swap_4_vora_decreasing2": [3, 2, 0, 1, 0, 3],
    "swap_4_vora_mid_bottleneck2": [3, 0, 1, 2, 0, 3],
    # for 5-repeater example
    "swap_5_asap": [1, 0, 0, 0, 0, 0, 1],
    "swap_5_baln": [3, 0, 1, 0, 2, 0, 3],  # need to specify exact doubling  => this is used in the vora paper
    "swap_5_baln2": [3, 0, 2, 0, 1, 0, 3],
    "swap_5_l2r": [5, 0, 1, 2, 3, 4, 5],
    "swap_5_r2l": [5, 4, 3, 2, 1, 0, 5],
    "swap_5_vora_uniform": [5, 0, 3, 1, 4, 2, 5],  # [3,0,1,0,2,0,3]  ~ baln
    "swap_5_vora_increasing": [5, 0, 3, 1, 4, 2, 5],  # [3,0,1,0,2,0,3] ~ baln
    "swap_5_vora_decreasing": [5, 2, 4, 1, 3, 0, 5],  # [3,0,2,0,1,0,3] ~ baln2
    "swap_5_vora_mid_bottleneck": [5, 0, 4, 2, 3, 1, 5],  # [3,0,2,0,1,0,3] ~ baln2
}


class ProactiveRoutingControllerApp(Application):
    """This is the centralized control plane app for Proactive Routing. Works with Proactive Forwarder on quantum nodes.
    A predefined swapping order has to be added in `swapping_settings` map and selected through the `swapping` parameter.
    """

    def __init__(self, swapping: str):
        """Args:
        swapping (str): swapping order to use for the S-D path, chosen from `swapping_settings` map.

        """
        super().__init__()
        self.net: QuantumNetwork
        """QN physical topology and classical topology"""
        self.own: Controller
        """controller node running this app"""

        try:
            self.swapping_order = swapping_settings[swapping]
        except KeyError:
            raise KeyError(f"{self.own}: Swapping {swapping} not configured")

        # self.add_handler(self.RecvClassicPacketHandler, [RecvClassicPacket])
        # self.server = HTTPServer(('', 8080), self.RequestHandler)
        # self.RequestHandler.test = self.test  # Pass test method to handler

    # class RequestHandler(BaseHTTPRequestHandler):
    #     def do_GET(self):
    #         self.send_response(200)
    #         self.end_headers()
    #         self.wfile.write(b"Test function executed")

    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        self.own = self.get_node(node_type=Controller)
        self.net = self.own.network

        # print("Starting server on port 8080...")
        # self.server.serve_forever()

        # install the test path on QNodes
        self.install_static_path()

    def install_static_path(self):
        """Install a static path between nodes "S" (source) and "D" (destination) of the network topology.
        It currently supports linear chain topology with a single S-D path.
        It computes a path between S and D and installs instructions with a predefined swapping order.

        This method performs the following:
            - Builds the network route table.
            - Locates the source ("S") and destination ("D") nodes in the network.
            - Computes shortest path between source and destination.
            DijkstraRouteAlgorithm is the default algorithm and uses hop counts at metric.
            - Verifies that the number of nodes in the computed path matches the expected number
            of swapping steps based on the configured swapping strategy.
            - Constructs per-channel memory allocation vectors (m_v) assuming buffer-space multiplexing
            defined by the memory capacity of node S.
            - Sends installation instructions to each node along the path using classical channels.

        Raises:
            Exception: If the computed route does not match the length expected by the current swapping configuration.

        Notes:
            - `self.swapping_order` provides the expected swapping instructions.
            - Multiplexing strategy is hardcoded as "B" (buffer-space).
            - Path instructions are transmitted over classical channels using the `"install_path"` command.
            - This method is primarily intended for test or static configuration scenarios, not dynamic routing.
            - Purification scheme has to be specified (hardcoded) in the function.

        """
        self.net.build_route()

        src = self.net.get_node("S")
        dst = self.net.get_node("D")

        route_result = self.net.query_route(src, dst)
        path_nodes = route_result[0][2]
        log.debug(f"{self.own}: Computed path: {path_nodes}")

        route = [n.name for n in path_nodes]

        if len(route) != len(self.swapping_order):
            raise Exception(
                f"{self.own}: Swapping {self.swapping_order} \
                does not correspond to computed route: {route}"
            )

        m_v = []
        src_capacity = self.net.get_node(path_nodes[0].name).get_memory().capacity
        for i in range(len(path_nodes) - 1):
            m_v.append(src_capacity)

        for qnode in path_nodes:
            instructions = {
                "route": route,
                "swap": self.swapping_order,
                "mux": "B",
                "m_v": m_v,
                # "purif": { 'S-R':1, 'R-D':1 },
                "purif": {},
            }

            cchannel = self.own.get_cchannel(qnode)
            classic_packet = ClassicPacket(
                msg={"cmd": "install_path", "path_id": 0, "instructions": instructions}, src=self.own, dest=qnode
            )
            cchannel.send(classic_packet, next_hop=qnode)
            log.debug(f"{self.own}: send {classic_packet.msg} to {qnode}")

    # def RecvClassicPacketHandler(self, node: Controller, event: Event):
    #     self.handle_request(event)

    # def handle_request(self, event: RecvClassicPacket):
    #     msg = event.packet.get()
    #     cchannel = event.cchannel

    #     from_node: Node = cchannel.node_list[0] \
    #         if cchannel.node_list[1] == self.own else cchannel.node_list[1]

    #     log.debug(f"{self.own}: recv {msg} from {from_node}")

    #     cmd = msg["cmd"]
    #     request_id = msg["request_id"]

    #     if cmd == "submit":
    #         # process new request submitted and send instructions to QNodes
    #         # can model processing time with events
    #         nodes_in_path = []
    #         for qnode in nodes_in_path:
    #             classic_packet = ClassicPacket(
    #                 msg={"cmd": "install_path", "request_id": request_id}, src=self.own, dest=qnode)
    #             cchannel.send(classic_packet, next_hop=qnode)
    #             log.debug(f"{self.own}: send {classic_packet.msg} to {qnode}")
    #     elif cmd == "withdraw":
    #         # remove request and send instructions to QNodes
    #         pass
    #     else:
    #         pass
