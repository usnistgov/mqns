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

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, RecvClassicPacket, classic_cmd_handler
from mqns.network.fw import RoutingController, RoutingPathStatic
from mqns.network.reactive.message import LinkStateMsg
from mqns.utils import log


class ReactiveRoutingController(ClassicCommandDispatcherMixin, RoutingController):
    """
    Centralized control plane for reactive routing.
    Works with ``ReactiveForwarder`` on quantum nodes.
    """

    def __init__(
        self,
        *,
        route: list[str] = ["S", "R", "D"],
        swap: list[int] | str,
    ):
        """
        Args:
            route: static path.
            swap: swapping policy.

        Note:
            This feature is in early stage.
            Currently it only installed a static path defined in ``route`` and ``swap``.
        """
        super().__init__()
        self.route = route
        self.swap = swap

        self.add_handler(self.handle_classic_command, RecvClassicPacket)

        self.ls_messages: list[LinkStateMsg] = []

    @override
    def install(self, node):
        super().install(node)

        self.requests = self.net.requests
        """Requests to satisfy in each routing phase."""

    @classic_cmd_handler("LS")
    def handle_ls(self, pkt: ClassicPacket, msg: LinkStateMsg):
        """
        Process received link_states from ReactiveForwarder.
        """

        if not self.node.timing.is_routing():  # should be in SYNC timing mode ROUTING phase
            log.debug(f"{self.node}: received LS message from {pkt.src} outside of ROUTING phase | {msg}")
            return True

        log.debug(f"{self.node.name}: received LS message from {pkt.src} | {msg}")

        self.ls_messages.append(msg)
        if len(self.ls_messages) == len(self.route):
            self.do_routing()
            self.ls_messages.clear()

        return True

    def do_routing(self):
        rpath = RoutingPathStatic(self.route, swap=self.swap)
        self.install_path(rpath)
