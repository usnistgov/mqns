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
from typing import cast, override

from mqns.entity.cchannel import ClassicPacket, classic_cmd_handler
from mqns.network.fw import PathInstructions, RoutingController
from mqns.network.network import (
    Request,
    RequestActiveEvent,
    RequestInactiveEvent,
    TimingModeSync,
    TimingPhase,
    sync_phase_handler,
)
from mqns.network.reactive.ctrl_tls import TopoLinkState
from mqns.network.reactive.message import LinkStateMsg
from mqns.simulator import event_handler, func_to_event
from mqns.utils import json_encodable, unwrap_cast


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


class ReactiveRoutingController(RoutingController):
    """
    Centralized control plane for reactive routing.
    Works with ``ReactiveForwarder`` on quantum nodes.

    This controller is only compatible with SYNC timing mode.
    It can automatically pick up requests added through ``QuantumNetwork``.

    The requests added to the network must not contain:

    * predefined ``RoutingPath``
    * multiplexing vector
    * purification scheme
    """

    def __init__(self):
        super().__init__()

        self.cnt = ReactiveRoutingControllerCounters()
        """
        Counters.
        """

        self._tls = TopoLinkState()
        """
        Topology link state.
        """

        self._reqs: dict[int, Request] = {}
        """
        Active requests.
        Key: request identifier.
        """

    @override
    def install(self, node):
        super().install(node)

        if self.node.timing.is_async():
            raise TypeError("ReactiveRoutingController only works with SYNC timing mode")
        timing = cast(TimingModeSync, self.node.timing)
        self.d_rtg = self.simulator.time(slot=timing.t_rtg.slot // 2)

    @event_handler
    def handle_request_active(self, event: RequestActiveEvent) -> None:
        req = event.req
        self.prepare_path(req)
        self._reqs[req.req_id] = req

    @event_handler
    def handle_request_inactive(self, event: RequestInactiveEvent) -> None:
        del self._reqs[event.req.req_id]

    @sync_phase_handler(TimingPhase.ROUTING, True)
    def sync_routing_enter(self) -> None:
        # Delete topology link state from previous time slot.
        self._tls.clear()
        # Wait half of ROUTING phase, then perform routing computation.
        self.simulator.sched(func_to_event(self.simulator.tc + self.d_rtg, self.do_routing))

    @classic_cmd_handler("LS")
    def handle_ls(self, pkt: ClassicPacket, msg: LinkStateMsg) -> None:
        """
        Process received link_states from ReactiveForwarder.
        """

        if not self.node.timing.is_routing():  # should be in SYNC timing mode ROUTING phase
            self.log_warning("received LS message from %s outside of ROUTING phase | %s", pkt.src.name, msg)
            return

        self.log_debug("received LS message from %s | %s", pkt.src.name, msg)
        self.cnt.n_ls += 1

        for entry in msg["ls"]:
            self._tls.add(entry)

    def do_routing(self) -> None:
        """
        Attempt to satisfy each active request with available entanglements.
        Repeat multiple rounds until no more requests can be satisfied.
        """
        satisfied = defaultdict[int, list[PathInstructions]](list)
        this_round_satisfied = True
        while this_round_satisfied:
            this_round_satisfied = False
            for req in self._reqs.values():
                inst = self._try_satisfy(req)
                if not inst:
                    continue

                satisfied[req.req_id].append(inst)
                self.cnt.n_satisfy += 1
                this_round_satisfied = True

        for req_id, insts in satisfied.items():
            self.path_insert(req_id, insts)

    def _try_satisfy(self, req: Request) -> PathInstructions | None:
        """
        Attempt to satisfy an active request with available entanglements.
        If the routing algorithm returns multiple routes, they will be tried in order.
        """
        insts = unwrap_cast(req.rp).compute_paths(self.route_ctx)
        for inst in insts:
            qubits = self._tls.try_consume(inst["route"])
            if not qubits:
                continue

            inst["reactive_qubits"] = qubits
            return inst
        return None
