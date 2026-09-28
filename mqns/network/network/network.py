#    Modified by Amar Abane for Multiverse Quantum Network Simulator
#    Date: 05/17/2025
#    Summary of changes: Adapted logic to support dynamic approaches.
#
#    This file is based on a snapshot of SimQN (https://github.com/QNLab-USTC/SimQN),
#    which is licensed under the GNU General Public License v3.0.
#
#    The original SimQN header is included below.


#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
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

from collections.abc import Collection, Iterable, MutableMapping
from typing import TYPE_CHECKING, Final, cast

from mqns.entity.base_channel import BaseChannel
from mqns.entity.cchannel import ClassicChannel
from mqns.entity.node import Controller, Node, QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement, EprTypeInput, parse_epr_type
from mqns.network.network.request import Request, RequestActiveEvent, RequestInactiveEvent
from mqns.network.network.timing import TimingMode, TimingModeAsync
from mqns.network.route import DijkstraRouteAlgorithm, RouteAlgorithm, RouteQueryResult
from mqns.network.topology import ClassicTopology, Topology
from mqns.simulator import Simulator, Time

if TYPE_CHECKING:
    from mqns.network.fw.routing import RoutingPath


def _save_channel[C: BaseChannel](d: MutableMapping[tuple[str, str], C], ch: C) -> None:
    if len(ch.node_list) != 2:
        return
    a, b = sorted(node.name for node in cast(list[Node], ch.node_list))
    d[a, b] = ch


def _get_channel[C: BaseChannel](d: dict[tuple[str, str], C], a: str, b: str) -> C:
    if a > b:
        a, b = b, a

    try:
        return d[a, b]
    except KeyError:
        raise LookupError(f"channel between {a} and {b} does not exist") from None


class QuantumNetwork:
    """QuantumNetwork includes quantum nodes, quantum and classical channels, arranged in a given topology."""

    timing: Final[TimingMode]
    """Network-wide application timing mode."""

    epr_type: Final[type[Entanglement]]
    """Network-wide entanglement type."""

    def __init__(
        self,
        topo: Topology | None = None,
        *,
        classic_topo: ClassicTopology | None = None,
        route: RouteAlgorithm[QNode, QuantumChannel] | None = None,
        timing: TimingMode = TimingModeAsync(),
        epr_type: EprTypeInput = None,
    ):
        """
        Args:
            topo: Topology builder.
            classic_topo: Classic topology parameter, passed to topology builder.
            route: Routing algorithm, defaults to dijkstra.
            timing: Network-wide application timing mode.
            epr_type: Network-wide entanglement type.
        """
        self.timing = timing
        self.epr_type = parse_epr_type(epr_type)
        assert getattr(self.epr_type, "__final__", False) is True, f"entanglement type {self.epr_type} must be marked @final"

        self._controller: Controller | None = None
        self._nodes: dict[str, QNode] = {}
        self._qchannels: dict[tuple[str, str], QuantumChannel] = {}
        self._cchannels: dict[tuple[str, str], ClassicChannel] = {}

        if topo is not None:
            self._populate_from_topo(topo, classic_topo)

        self.route: RouteAlgorithm = route or DijkstraRouteAlgorithm()
        """Routing algorithm."""

        self.requests: list[Request] = []
        """Requested end-to-end entanglements."""

    def _populate_from_topo(self, topo: Topology, classic_topo: ClassicTopology | None) -> None:
        nodes, qchannels = topo.build()
        if classic_topo is not None:
            cchannels = topo.add_cchannels(classic_topo=classic_topo, nl=nodes, ll=qchannels)
        else:
            cchannels = topo.add_cchannels()

        for node in nodes:
            self.add_node(node)
        for ch in qchannels:
            self.add_qchannel(ch)
        for ch in cchannels:
            self.add_cchannel(ch)

        if topo.controller:
            self.set_controller(topo.controller)

    def install(self, simulator: Simulator) -> None:
        """
        Install all nodes (including channels, memories and applications) in this network.

        Args:
            simulator: The simulator.
        """
        self.simulator = simulator
        """Simulator instance."""

        self.timing.install(self)

        for node in self.all_nodes:
            node.install(simulator)

        for req in self.requests:
            self._install_request(req)

    @property
    def controller(self) -> Controller | None:
        """Retrieve the controller node, if present."""
        return self._controller

    def set_controller(self, controller: Controller) -> None:
        """Set the controller node."""
        Simulator.ensure_not_installed_to(self)
        assert self._controller is None, "controller exists"
        self._controller = controller
        controller.add_network(self)

    @property
    def nodes(self) -> Collection[QNode]:
        """Retrieve a collection of quantum nodes."""
        return self._nodes.values()

    def get_node(self, name: str) -> QNode:
        """
        Retrieve a quantum node by name.

        Raises:
            LookupError: Node does not exist.
        """
        try:
            return self._nodes[name]
        except KeyError:
            raise LookupError(f"node {name} does not exist") from None

    def add_node(self, node: QNode) -> None:
        """
        Add a QNode into this network.
        """
        Simulator.ensure_not_installed_to(self)
        assert node.name not in self._nodes, f"duplicate node name {node.name}"
        self._nodes[node.name] = node
        node.add_network(self)

    @property
    def all_nodes(self) -> Iterable[Node]:
        """Iterate over all quantum nodes and the controller."""
        yield from self.nodes
        if self._controller is not None:
            yield self._controller

    @property
    def qchannels(self) -> Collection[QuantumChannel]:
        """Retrieve a collection of quantum channels."""
        return self._qchannels.values()

    def get_qchannel(self, a: str, b: str) -> QuantumChannel:
        """
        Retrieve a quantum channel by node names.

        Raises:
            LookupError: channel does not exist.
        """
        return _get_channel(self._qchannels, a, b)

    def add_qchannel(self, qchannel: QuantumChannel) -> None:
        """
        Add a quantum channel into this network.
        """
        Simulator.ensure_not_installed_to(self)
        _save_channel(self._qchannels, qchannel)

    @property
    def cchannels(self) -> Collection[ClassicChannel]:
        """Retrieve a collection of classic channels."""
        return self._cchannels.values()

    def get_cchannel(self, a: str, b: str) -> ClassicChannel:
        """
        Retrieve a classic channel by node names.

        Raises:
            LookupError: channel does not exist.
        """
        return _get_channel(self._cchannels, a, b)

    def add_cchannel(self, cchannel: ClassicChannel) -> None:
        """
        Add a classic channel into this network.
        """
        Simulator.ensure_not_installed_to(self)
        _save_channel(self._cchannels, cchannel)

    def build_route(self) -> None:
        """Build static route tables."""
        self.route.build(list(self.nodes), self.qchannels)

    def query_route(self, src: str, dst: str, /, error_on_empty=True) -> list[RouteQueryResult[QNode]]:
        """
        Query the routing algorithm.

        Args:
            src: Source node.
            dst: Destination node.
            error_on_empty: If true, raise RuntimeError if there's no route.

        Returns:
            List of route paths, sorted by priority.
        """
        routes = self.route.query(self._nodes[src], self._nodes[dst])
        if error_on_empty and not routes:
            raise RuntimeError(f"no route from {src} to {dst}")
        return routes

    def add_request(self, *args: "Request|RoutingPath") -> None:
        """
        Add one or more requests to the network.
        """
        reqs = [(req if isinstance(req, Request) else Request(req)) for req in args]
        self.requests.extend(reqs)

        if hasattr(self, "simulator"):
            for req in reqs:
                self._install_request(req)

    def _install_request(self, req: Request) -> None:
        req.active_since = Time.from_time_or_sec(req._active_since_input, accuracy=self.simulator.accuracy)
        req.active_until = Time.from_time_or_sec(req._active_until_input, accuracy=self.simulator.accuracy)

        t_enter = self.simulator.tc if req.active_since is Time.MIN else req.active_since
        self.simulator.sched(RequestActiveEvent(self.controller, req, t=t_enter))

        if req.active_until is not Time.MAX:
            self.simulator.sched(event := RequestInactiveEvent(self.controller, req, t=req.active_until))
            req.inactive_event.set(event)
