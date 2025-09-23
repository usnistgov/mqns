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


import numpy as np
from scipy.sparse.csgraph import yen

from mqns.network.route.route import ChannelT, MetricFunc, NodeT, RouteImpl, make_csr


class YenRouteAlgorithm(RouteImpl[NodeT, ChannelT]):
    """This is the Yen's algorithm implementation"""

    def __init__(self, name: str = "yen", metric_func: MetricFunc | None = None, k_paths: int = 3) -> None:
        """
        Args:
            name: Name of the routing algorithm (default: "yen").
            metric_func: Function returning the metric (weight) for each channel.
                Defaults to a constant function m(l) = 1.
            k_paths: Number of shortest paths to compute for each (src, dst) pair.
                Default is 3.
        """
        self.name = name
        self.k_paths = k_paths
        self.route_table: dict[NodeT, dict[NodeT, list[tuple[float, list[NodeT]]]]] = {}

        if metric_func is None:
            self.metric_func = lambda _: 1  # hop count
            self.unweighted = True
        else:
            self.metric_func = metric_func
            self.unweighted = False

    def build(self, nodes: list[NodeT], channels: list[ChannelT]):
        """
        Build the routing table using SciPy's csgraph Yen's algorithm on a CSR adjacency.
        Up to self.k_paths pahrs are computred for each (src, dst) pair.

        Args:
            nodes: a list of quantum nodes or classic nodes
            channels: a list of quantum channels or classic channels
        """

        # build adjacency matrix
        csr_adj = make_csr(nodes, channels, self.metric_func)

        # Helper: reconstruct one path given a predecessor row from yen()
        def _reconstruct_path(pred_row: np.ndarray, src_idx: int, dst_idx: int) -> list[NodeT]:
            path_idx: list[int] = []
            u = dst_idx
            # SciPy uses -9999 as sentinel for "no predecessor"
            while u != -9999:
                path_idx.append(u)
                if u == src_idx:
                    break
                u = int(pred_row[u])
            # If we didn’t reach src, path is invalid
            if not path_idx or path_idx[-1] != src_idx:
                return []

            return [nodes[i] for i in path_idx]

        self.route_table.clear()

        for s_idx, src in enumerate(nodes):
            self.route_table[src] = {}
            for t_idx, dst in enumerate(nodes):
                if s_idx == t_idx:
                    continue

                # Run Yen's K-shortest for this pair
                # unweighted=True -> hop count; directed=False for undirected topologies
                dist_array, predecessors = yen(
                    csgraph=csr_adj,
                    source=s_idx,
                    sink=t_idx,
                    K=self.k_paths,
                    directed=False,
                    return_predecessors=True,
                    unweighted=self.unweighted,
                )

                # dist_array shape: (M,), predecessors shape: (M, n), with M <= K
                if dist_array.size == 0:
                    self.route_table[src][dst] = []
                    continue

                route_list: list[tuple[float, list[NodeT]]] = []
                for i in range(dist_array.shape[0]):
                    cost = float(dist_array[i])
                    pred_row = predecessors[i]
                    node_path = _reconstruct_path(pred_row, s_idx, t_idx)
                    route_list.append((cost, node_path))

                self.route_table[src][dst] = route_list

    def query(self, src: NodeT, dest: NodeT) -> list[tuple[float, NodeT, list[NodeT]]]:
        paths = self.route_table.get(src, {}).get(dest, [])
        results = []
        for metric, path in paths:
            if len(path) > 1:
                results.append((metric, path[1], list(reversed(path))))
        return results
