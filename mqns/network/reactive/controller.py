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

from collections import defaultdict
from itertools import pairwise
from typing import cast, override

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, RecvClassicPacket, classic_cmd_handler
from mqns.network.fw import RoutingController, RoutingPathStatic, SwapPolicy
from mqns.network.network import Request, TimingModeSync, TimingPhase, TimingPhaseEvent
from mqns.network.reactive.message import LinkStateMsg
from mqns.simulator import func_to_event
from mqns.utils import json_encodable, log


@json_encodable
class ReactiveRoutingControllerCounters:
    """Counters related to ``ReactiveRoutingController``."""

    def __init__(self):
        self.n_ls = 0
        """How many link-state message arrived."""
        self.n_satisfy = 0
        """How many requests satisfied."""

    def __repr__(self) -> str:
        return f"ls={self.n_ls} satisfy={self.n_satisfy}"


class ReactiveRoutingController(ClassicCommandDispatcherMixin, RoutingController):
    """
    Centralized control plane for reactive routing.
    Works with ``ReactiveForwarder`` on quantum nodes.

    This controller is only compatible with SYNC timing mode.
    It can automatically pick up requests from ``QuantumNetwork.requests`` list.
    """

    def __init__(
        self,
        *,
        swap: SwapPolicy = "asap",
    ):
        """
        Args:
            swap: Swapping policy applied to all paths.
        """
        super().__init__()

        self.swap: SwapPolicy = swap

        self.cnt = ReactiveRoutingControllerCounters()
        """
        Counters.
        """

        self.add_handler(self.handle_classic_command, RecvClassicPacket)
        self.add_handler(self.handle_sync_phase, TimingPhaseEvent)

        self._tls = defaultdict[tuple[str, str], set[str]](set)
        """
        Topology link state.

        Key: node names, sorted.
        Value: entanglement reservation keys.
        """

    @override
    def install(self, node):
        super().install(node)

        if self.node.timing.is_async():
            raise TypeError("ReactiveRoutingController only works with SYNC timing mode")
        self.timing = cast(TimingModeSync, self.node.timing)
        self.d_rtg = self.simulator.time(time_slot=self.timing.t_rtg.time_slot // 2)

    def handle_sync_phase(self, event: TimingPhaseEvent):
        match event.action:
            case TimingPhase.ROUTING, True:
                self._tls.clear()
                self.simulator.add_event(func_to_event(self.simulator.tc + self.d_rtg, self.do_routing))

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

        for entry in msg["ls"]:
            n0, n1 = entry["node"], entry["neighbor"]
            if n0 < n1:
                self._tls[(n0, n1)].add(entry["qubit"])

        return True

    def do_routing(self):
        """
        Attempt to satisfy each active request in ``QuantumNetwork.requests`` list with available entanglements.
        Repeat multiple rounds until no more requests can be satisfied.
        """
        some_satisfied = True
        while some_satisfied:
            some_satisfied = False
            for req in self.net.requests:
                if self._try_satisfy(req):
                    self.cnt.n_satisfy += 1
                    some_satisfied = True

    def _try_satisfy(self, req: Request) -> bool:
        """
        Attempt to satisfy an active request with available entanglements.
        If the routing algorithm returns multiple routes, they will be tried in order.
        """
        route_result = self.net.query_route(req.src, req.dst)
        for _, _, route_nodes in route_result:
            route = [node.name for node in route_nodes]

            if (qubits := self._try_consume(route)) is None:
                continue

            self.install_path(RoutingPathStatic(route, m_v=qubits, swap=self.swap))
            return True

        return False

    def _try_consume(self, route: list[str]) -> list[str] | None:
        """
        Attempt to match a computed route with available entanglements.
        The entanglements are removed from ``self._tls`` only if every link along the path has an entanglement.
        """
        link_etgs: list[set[str]] = []

        for n0, n1 in pairwise(route):
            etgs = self._tls.get((n0, n1) if n0 < n1 else (n1, n0))
            if etgs is None or len(etgs) == 0:
                return None
            link_etgs.append(etgs)

        return [etgs.pop() for etgs in link_etgs]
