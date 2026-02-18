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

from typing import cast, overload

from mqns.entity.base_channel import BaseChannel
from mqns.entity.cchannel import ClassicChannel
from mqns.entity.node import Controller, Node, QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement, WernerStateEntanglement
from mqns.network.network.request import Request
from mqns.network.network.timing import TimingMode, TimingModeAsync
from mqns.network.route import DijkstraRouteAlgorithm, RouteAlgorithm, RouteQueryResult
from mqns.network.topology import ClassicTopology, Topology
from mqns.simulator import Simulator
from mqns.utils import rng


def _save_channel[C: BaseChannel](l: list[C], d: dict[tuple[str, str], C], ch: C):
    l.append(ch)
    if len(ch.node_list) != 2:
        return
    a, b = sorted((node.name for node in cast(list[Node], ch.node_list)))
    d[(a, b)] = ch


def _get_channel[C: BaseChannel](l: list[C], d: dict[tuple[str, str], C], q: tuple[str, ...]):
    if len(q) == 1:
        name = q[0]
        for ch in l:
            if ch.name == name:
                return ch
        raise IndexError(f"channel {name} does not exist")

    a, b = sorted(q)
    try:
        return d[(a, b)]
    except KeyError:
        raise IndexError(f"channel between {a} and {b} does not exist")


class QuantumNetwork:
    """QuantumNetwork includes quantum nodes, quantum and classical channels, arranged in a given topology"""

    def __init__(
        self,
        topo: Topology | None = None,
        *,
        classic_topo: ClassicTopology | None = None,
        route: RouteAlgorithm[QNode, QuantumChannel] | None = None,
        timing: TimingMode = TimingModeAsync(),
        epr_type: type[Entanglement] = WernerStateEntanglement,
    ):
        """
        Args:
            topo: topology builder.
            classic_topo: classic topology parameter, passed to topology builder.
            route: routing algorithm, defaults to dijkstra.
            timing: network-wide application timing mode.
            epr_type: network-wide entanglement type.
        """
        assert getattr(epr_type, "__final__", False) is True, f"entanglement type {epr_type} must be marked @final"

        self.timing = timing
        """Network-wide application timing mode."""
        self.epr_type = epr_type
        """Network-wide entanglement type."""

        self.controller: Controller | None = None
        """Controller node."""
        self.nodes: list[QNode] = []
        """List of quantum nodes."""
        self._node_by_name: dict[str, QNode] = {}
        self.qchannels: list[QuantumChannel] = []
        """List of quantum channels."""
        self._qchannel_by_ends: dict[tuple[str, str], QuantumChannel] = {}
        self.cchannels: list[ClassicChannel] = []
        """List of classic channels."""
        self._cchannel_by_ends: dict[tuple[str, str], ClassicChannel] = {}

        if topo is not None:
            self._populate_from_topo(topo, classic_topo)

        self.route: RouteAlgorithm = DijkstraRouteAlgorithm() if route is None else route
        """Routing algorithm."""

        self.requests: list[Request] = []
        """Requested end-to-end entanglements."""

    def _populate_from_topo(self, topo: Topology, classic_topo: ClassicTopology | None):
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

    def _ensure_not_installed(self) -> None:
        """
        Assert that this entity has not been installed into a simulator.
        """
        assert not hasattr(self, "simulator"), "function only available prior to self.install()"

    def install(self, simulator: Simulator):
        """
        Install all nodes (including channels, memories and applications) in this network

        Args:
            simulator: the simulator

        """
        self.simulator = simulator
        """Simulator instance."""

        self.all_nodes: list[Node] = []
        """A collection of quantum nodes and the controller (if present)."""
        self.all_nodes += self.nodes
        if self.controller:
            self.all_nodes.append(self.controller)

        for node in self.all_nodes:
            node.install(simulator)
        self.timing.install(self)

    def add_node(self, node: QNode):
        """
        Add a QNode into this network.
        """
        self._ensure_not_installed()
        assert node.name not in self._node_by_name, f"duplicate node name {node.name}"
        self.nodes.append(node)
        self._node_by_name[node.name] = node
        node.add_network(self)

    def get_node(self, name: str) -> QNode:
        """
        Get QNode by name.

        Raises:
            IndexError - node does not exist.
        """
        try:
            return self._node_by_name[name]
        except KeyError:
            raise IndexError(f"node {name} does not exist")

    def set_controller(self, controller: Controller):
        """
        Set the controller of this network.
        """
        self._ensure_not_installed()
        self.controller = controller
        controller.add_network(self)

    def get_controller(self) -> Controller:
        """
        Get the Controller of this network.

        Raises:
            IndexError - controller does not exist.
        """
        if self.controller is None:
            raise IndexError("network does not have a controller")
        return self.controller

    def add_qchannel(self, qchannel: QuantumChannel):
        """
        Add a QuantumChannel into this network.
        """
        self._ensure_not_installed()
        _save_channel(self.qchannels, self._qchannel_by_ends, qchannel)

    @overload
    def get_qchannel(self, name: str, /) -> QuantumChannel:
        """
        Retrieve QuantumChannel by name.

        Raises:
            IndexError - channel does not exist.
        """
        pass

    @overload
    def get_qchannel(self, a: str, b: str, /) -> QuantumChannel:
        """
        Retrieve QuantumChannel by node names.

        Raises:
            IndexError - channel does not exist.
        """
        pass

    def get_qchannel(self, *q: str) -> QuantumChannel:
        return _get_channel(self.qchannels, self._qchannel_by_ends, q)

    def add_cchannel(self, cchannel: ClassicChannel):
        """
        Add a ClassicChannel into this network.
        """
        self._ensure_not_installed()
        _save_channel(self.cchannels, self._cchannel_by_ends, cchannel)

    @overload
    def get_cchannel(self, name: str, /) -> ClassicChannel:
        """
        Retrieve ClassicalChannel by name.

        Raises:
            IndexError - channel does not exist.
        """
        pass

    @overload
    def get_cchannel(self, a: str, b: str, /) -> ClassicChannel:
        """
        Retrieve ClassicalChannel by node names.

        Raises:
            IndexError - channel does not exist.
        """
        pass

    def get_cchannel(self, *q: str) -> ClassicChannel:
        return _get_channel(self.cchannels, self._cchannel_by_ends, q)

    def build_route(self):
        """Build static route tables for each nodes"""
        self.route.build(self.nodes, self.qchannels)

    def query_route(self, src: QNode, dest: QNode) -> list[RouteQueryResult[QNode]]:
        """Query the metric, nexthop and the path

        Args:
            src: the source node
            dest: the destination node

        Returns:
            A list of route paths. The result should be sorted by the priority.
            The element is a tuple containing: metric, the next-hop and the whole path.

        """
        return self.route.query(src, dest)

    def add_request(self, src: QNode, dst: QNode, attr: dict = {}):
        """
        Add a request (src, dst) pair to the network.

        The request is placed in `self.requests` list.
        The scenario must manually pass these requests to relevant applications (e.g. ProactiveRoutingController).

        Args:
            src: the source node
            dst: the destination node
            attr: other attributions
        """
        req = Request(src, dst, attr)
        self.requests.append(req)

    def random_requests(
        self,
        n: int,
        *,
        clear=True,
        allow_overlay=False,
        min_hops=1,
        max_hops=10,
        attr: dict | None = None,
        forbid_endpoint_internal=True,  # reject endpoint-vs-internal conflicts
    ):
        """
        Generate random (src, dst) pairs requests.

        The requests are placed in `self.requests` list.
        The scenario must manually pass these requests to relevant applications (e.g. ProactiveRoutingController).

        Args:
            n: number of requests to generate
            clear: if True, clear existing requests in `self.requests`
            allow_overlay: allow nodes to be the source or destination in multiple requests
            min_hops: minimum number of hops (inclusive)
            max_hops: maximum number of hops (inclusive)
            attr: request attributes
            forbid_endpoint_internal: if True, eliminate requests that
                would fail the rank-based endpoint-vs-internal check in SWAP-ASAP.
        """
        attr = {} if attr is None else attr
        used_nodes: list[int] = []
        nnodes = len(self.nodes)

        if n < 1:
            raise ValueError("number of requests should be larger than 1")
        if not allow_overlay and n * 2 > nnodes:
            raise ValueError("Too many requests")

        if clear:
            self.requests.clear()

        # Track accepted paths
        accepted_paths: list[dict] = []  # each: {"endpoints": set, "edges": set}

        def to_meta(path_nodes: list[QNode]) -> dict:
            endpoints = {path_nodes[0].name, path_nodes[-1].name}
            edges = {(path_nodes[i].name, path_nodes[i + 1].name) for i in range(len(path_nodes) - 1)}
            return {"endpoints": endpoints, "edges": edges}

        def violates_endpoint_internal(candidate_meta: dict) -> bool:
            cend = candidate_meta["endpoints"]
            cedges = candidate_meta["edges"]
            for meta in accepted_paths:
                pend = meta["endpoints"]
                pedges = meta["edges"]
                shared = cedges & pedges
                if not shared:
                    continue
                for u, v in shared:
                    # one path treats node as endpoint, other as internal
                    if ((u in cend) != (u in pend)) or ((v in cend) != (v in pend)):
                        return True
            return False

        for _ in range(n):
            while True:
                src_idx = rng.integers(0, nnodes, dtype=int)
                dst_idx = rng.integers(0, nnodes, dtype=int)
                if src_idx == dst_idx:
                    continue
                if not allow_overlay and (src_idx in used_nodes or dst_idx in used_nodes):
                    continue

                src = self.nodes[src_idx]
                dst = self.nodes[dst_idx]
                route_result = self.query_route(src, dst)
                if not route_result:
                    continue

                hops, _, path_nodes = route_result[0]
                if not (min_hops <= hops <= max_hops):
                    continue

                if forbid_endpoint_internal:
                    meta = to_meta(path_nodes)
                    if violates_endpoint_internal(meta):
                        continue
                    accepted_paths.append(meta)

                # Accept
                if not allow_overlay:
                    used_nodes.extend([src_idx, dst_idx])

                self.add_request(src, dst, attr)
                break
