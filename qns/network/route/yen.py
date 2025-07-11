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


from typing import Callable

import networkx as nx

from qns.network.route.route import ChannelT, NetworkRouteError, NodeT, RouteImpl

MetricFunc = Callable[[ChannelT], float]


class YenRouteAlgorithm(RouteImpl[NodeT, ChannelT]):
    """Yen's algorithm using NetworkX's k-shortest simple paths."""

    def __init__(self, name: str = "yen", metric_func: MetricFunc = lambda _: 1, k_paths: int = 3) -> None:
        self.name = name
        self.metric_func = metric_func
        self.k_paths = k_paths
        self.route_table: dict[NodeT, dict[NodeT, list[tuple[float, list[NodeT]]]]] = {}

    def build(self, nodes: list[NodeT], channels: list[ChannelT]):
        # Build a NetworkX graph
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node)

        for ch in channels:
            if len(ch.node_list) != 2:
                raise NetworkRouteError("broken link")
            a, b = ch.node_list
            w = self.metric_func(ch)
            G.add_edge(a, b, weight=w)
            G.add_edge(b, a, weight=w)  # assume undirected links

        for src in nodes:
            self.route_table[src] = {}
            for dst in nodes:
                if src == dst:
                    continue
                try:
                    paths = list(nx.shortest_simple_paths(G, src, dst, weight="weight"))
                except nx.NetworkXNoPath:
                    self.route_table[src][dst] = []
                    continue

                route_list = []
                for path in paths[: self.k_paths]:
                    cost = sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))
                    route_list.append((cost, path))
                self.route_table[src][dst] = route_list

    def query(self, src: NodeT, dest: NodeT) -> list[tuple[float, NodeT, list[NodeT]]]:
        paths = self.route_table.get(src, {}).get(dest, [])
        results = []
        for metric, path in paths:
            if len(path) > 1:
                results.append((metric, path[1], path))
        return results
