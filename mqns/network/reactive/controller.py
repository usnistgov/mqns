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
from mqns.network.fw import RoutingController, RoutingPathStatic, SwapSequenceInput
from mqns.network.reactive.message import LinkStateMsg
from mqns.utils import json_encodable, log


@json_encodable
class ReactiveRoutingControllerCounters:
    """Counters related to ``ReactiveRoutingController``."""

    def __init__(self):
        self.n_ls = 0
        """How many link-state message arrived."""
        self.n_decision = 0
        """How many routing decisions sent."""

    def __repr__(self) -> str:
        return f"ls={self.n_ls} decision={self.n_decision}"


class ReactiveRoutingController(ClassicCommandDispatcherMixin, RoutingController):
    """
    Centralized control plane for reactive routing.
    Works with ``ReactiveForwarder`` on quantum nodes.
    """

    def __init__(
        self,
        *,
        route: list[str] = ["S", "R", "D"],
        swap: SwapSequenceInput = "asap",
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
        self.swap: SwapSequenceInput = swap

        self.ls_messages: list[LinkStateMsg] = []
        """
        Received but unprocessed link-state messages.
        """

        self.cnt = ReactiveRoutingControllerCounters()
        """
        Counters.
        """

        self.add_handler(self.handle_classic_command, RecvClassicPacket)

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
        self.cnt.n_ls += 1
        self.ls_messages.append(msg)

        if len(self.ls_messages) == len(self.route):
            self.do_routing()
            self.ls_messages.clear()

        return True

    def do_routing(self):
        rpath = RoutingPathStatic(self.route, swap=self.swap)
        self.install_path(rpath)

    @override
    def install_path(self, rp):
        self.cnt.n_decision += 1
        super().install_path(rp)
