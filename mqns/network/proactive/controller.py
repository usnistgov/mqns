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

from typing_extensions import override

from mqns.entity.cchannel import ClassicPacket
from mqns.entity.node import Application, Controller, Node
from mqns.network.proactive.message import InstallPathMsg, PathInstructions, UninstallPathMsg
from mqns.network.proactive.routing import RoutingPath
from mqns.simulator import Simulator
from mqns.utils import log


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

    @override
    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        self.own = self.get_node(node_type=Controller)
        self.net = self.own.network
        self.next_req_id = 0
        self.next_path_id = 0

        # install pre-requested paths on QNodes
        self.net.build_route()
        for rp in self.paths:
            self.install_path(rp)

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

    def uninstall_path(self, rp: RoutingPath):
        """
        Compute routing path(s) and send uninstall commands to nodes.
        """
        assert rp.req_id >= 0
        assert rp.path_id >= 0

        for path_id_add, instructions in enumerate(rp.compute_paths(self.net)):
            self._send_instructions(rp.path_id + path_id_add, instructions, uninstall=True)

    def _send_instructions(self, path_id: int, instructions: PathInstructions, *, uninstall=False):
        verb, msg = (
            ("uninstall", UninstallPathMsg(cmd="uninstall_path", path_id=path_id))
            if uninstall
            else ("install", InstallPathMsg(cmd="install_path", path_id=path_id, instructions=instructions))
        )

        for node_name in instructions["route"]:
            qnode = self.net.get_node(node_name)
            self.own.get_cchannel(qnode).send(ClassicPacket(msg, src=self.own, dest=qnode), next_hop=qnode)
            log.debug(f"{self.own}: {verb} path #{path_id} at {qnode}: {instructions}")
