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

from typing import override

from mqns.entity.cchannel import ClassicPacket, RecvClassicPacket
from mqns.entity.node import Application, Controller
from mqns.entity.base_channel import ChannelT, NodeT
from mqns.network.route.dijkstra import DijkstraRouteAlgorithm
from mqns.network.route.route import RouteAlgorithm
from mqns.network.proactive.message import InstallPathMsg, PathInstructions, UninstallPathMsg
from mqns.network.proactive.routing import RoutingPath, RoutingPathStatic
from mqns.utils import log


class ReactiveRoutingController(Application[Controller]):
    """
    Centralized control plane app for Reactive Routing.
    Works with ReactiveForwarder on quantum nodes.
    """

    def __init__(
        self,
        swap: list[int] | str,
    ):
        """
        Args:
            swap: swapping policy to apply to all paths.
        """
        super().__init__()
        self.swap = swap
        
        self.add_handler(self.RecvClassicPacketHandler, RecvClassicPacket)
        
        self.ls_messages = []

    @override
    def install(self, node):
        self._application_install(node, Controller)
        self.net = self.node.network
        self.requests = self.net.requests   # requests to satisfy in each routing phase
        self.next_req_id = 0
        self.next_path_id = 0


    def RecvClassicPacketHandler(self, event: RecvClassicPacket) -> bool:
        """
        Process a received classical packet.
        The packet is expected to contain link_states from nodes and received during the ROUTING phase.

        This method recognizes a message that is a dict with "ls" key with known value.

        Returns False for unrecognized message types, which allows the packet to go to the next application.
        """
        packet = event.packet
        msg = packet.get()
        if not (isinstance(msg, dict) and "ls" in msg):
            return False
        if not self.node.timing.is_routing():  # should be in SYNC timing mode ROUTING phase
            log.debug(f"{self.node}: received ls message from {packet.src} outside of ROUTING phase | {msg}")
            return False

        log.debug(f"{self.node.name}: received LS message from {packet.src} | {msg}")
        
        self.ls_messages.append(msg)
        if len(self.ls_messages) == 3:
            self.do_routing()
            self.ls_messages = []
    
        return True


    # Handle INTERNAL/ROUTING phase signal:
    # build logical topology
    # compute paths for requests
    # create path = ReactiveRoutingPath(RoutingPath) for each path
    # path.compute_paths (i.e., instructions)
    # self.instructions
    # Instructions format may be adapted to ReactiveRouting

    def do_routing(self):
        rpath = RoutingPathStatic(["S", "R", "D"], swap=self.swap)
        self.install_path(rpath)
        

    # Try to reuse this in Reactive!!!
    def install_path(self, rp: RoutingPath):
        """
        Compute routing path(s) and send install commands to nodes.
        """
        if rp.req_id < 0:
            rp.req_id = self.next_req_id
        self.next_req_id = max(self.next_req_id, rp.req_id + 1)

        if rp.path_id < 0:
            rp.path_id = self.next_path_id

        for path_id_add, instructions in enumerate(rp.compute_paths(self.net)):
            path_id = rp.path_id + path_id_add
            self.next_path_id = max(self.next_path_id, path_id + 1)
            self._send_instructions(path_id, instructions)

    def _send_instructions(self, path_id: int, instructions: PathInstructions, *, uninstall=False):
        verb, msg = (
            ("uninstall", UninstallPathMsg(cmd="uninstall_path", path_id=path_id))
            if uninstall
            else ("install", InstallPathMsg(cmd="install_path", path_id=path_id, instructions=instructions))
        )

        for node_name in instructions["route"]:
            qnode = self.net.get_node(node_name)
            self.node.get_cchannel(qnode).send(ClassicPacket(msg, src=self.node, dest=qnode), next_hop=qnode)
            log.debug(f"{self.node}: {verb} path #{path_id} at {qnode}: {instructions}")
