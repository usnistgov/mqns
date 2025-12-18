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

from typing import Any, override

import numpy as np
from scipy.sparse.csgraph import dijkstra

from mqns.entity.base_channel import ChannelT, NodeT
from mqns.network.route.route import MetricFunc, RouteAlgorithm, RouteQueryResult, make_csr


class DijkstraRouteAlgorithm(RouteAlgorithm[NodeT, ChannelT]):
    """
    Dijkstra algorithm.

    This is implemented with SciPy's csgraph Dijkstra on a CSR adjacency.
    """

    @override
    def __init__(self, name="dijkstra", metric_func: MetricFunc | None = None) -> None:
        """
        Args:
            name: Name of the routing algorithm (default: "dijkstra").
            metric_func: Function returning the metric (weight) for each channel.
                Defaults to a constant function m(l) = 1.
        """
        super().__init__(name, metric_func)
        self.route_table: dict[NodeT, dict[NodeT, tuple[float, list[NodeT]]]] = {}

    @override
    def build(self, nodes: list[NodeT], channels: list[ChannelT]):
        # build adjacency matrix
        csr_adj = make_csr(nodes, channels, self.metric_func)

        # unweighted=True -> hop count; directed=False for undirected topologies
        dist, preds = dijkstra(
            csr_adj,
            directed=False,
            unweighted=self.unweighted,
            return_predecessors=True,
        )

        # Reconstruct path helper
        def _reconstruct_path(src_idx: int, dst_idx: int) -> list[NodeT]:
            # Backtrack from dst to src using predecessors
            path_idx: list[int] = []
            u = dst_idx
            while u not in (-9999, src_idx):
                path_idx.append(u)
                u = preds[src_idx, u]
            path_idx.append(src_idx)
            return [nodes[i] for i in path_idx]

        self.route_table.clear()

        # For each source node, create the per-destination entry
        for src_idx, src_node in enumerate(nodes):
            dest_entry: dict[NodeT, Any] = {}

            for dst_idx, dst_node in enumerate(nodes):
                if src_idx == dst_idx:
                    # Source to itself
                    dest_entry[dst_node] = [0.0, [dst_node]]
                    continue

                hop = dist[src_idx, dst_idx]
                if np.isinf(hop):  # Unreachable
                    dest_entry[dst_node] = [np.inf, [dst_node]]
                else:
                    path_nodes = _reconstruct_path(src_idx, dst_idx)
                    dest_entry[dst_node] = [hop, path_nodes]

            self.route_table[src_node] = dest_entry

    @override
    def query(self, src: NodeT, dst: NodeT) -> list[RouteQueryResult]:
        ls = self.route_table.get(src, None)
        if ls is None:
            return []
        le = ls.get(dst, None)
        if le is None:
            return []
        try:
            metric, path = le
            path = path.copy()
            path.reverse()
            if len(path) <= 1 or np.isinf(metric):  # unreachable
                next_hop = None
                return []
            else:
                next_hop = path[1]
                return [RouteQueryResult(metric, next_hop, path)]
        except Exception:
            return []
