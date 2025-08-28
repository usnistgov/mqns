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
from qns.network.proactive.message import InstallPathMsg
from qns.network.proactive.routing import RoutingPath
from qns.simulator import Simulator
from qns.utils import log


class ProactiveRoutingController(Application):
    """
    Centralized control plane app for Proactive Routing.
    Works with Proactive Forwarder on quantum nodes.
    """

    def __init__(
        self,
        paths: RoutingPath | list[RoutingPath] | None = None,
    ):
        """
        Args:
            paths: routing path(s) to be automatically installed when the application is initiated.
        """
        super().__init__()
        self.paths = [] if not paths else paths if isinstance(paths, list) else [paths]

    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        self.own = self.get_node(node_type=Controller)
        self.net = self.own.network
        self.next_req_id = 0
        self.next_path_id = 0

        # install the test path on QNodes
        self.net.build_route()
        for rp in self.paths:
            self.install_path(rp)

    def install_path(self, rp: RoutingPath):
        """
        Compute routing path(s) and install onto nodes.
        """
        if rp.req_id < 0:
            rp.req_id = self.next_req_id
        self.next_req_id = max(self.next_req_id, rp.req_id + 1)

        if rp.path_id < 0:
            rp.path_id = self.next_path_id

        for path_id_add, instructions in enumerate(rp.compute_paths(self.net)):
            path_id = rp.path_id + path_id_add
            self.next_path_id = max(self.next_path_id, path_id + 1)

            route = instructions["route"]
            for node_name in route:
                qnode = self.net.get_node(node_name)
                msg: "InstallPathMsg" = {"cmd": "install_path", "path_id": path_id, "instructions": instructions}

                cchannel = self.own.get_cchannel(qnode)
                cchannel.send(ClassicPacket(msg, src=self.own, dest=qnode), next_hop=qnode)
                log.debug(f"{self.own}: send {msg} to {qnode}")
