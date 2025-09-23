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

from collections.abc import Callable
from typing import Generic

import numpy as np
from scipy.sparse import csr_matrix

from mqns.entity import ChannelT, NodeT

MetricFunc = Callable[[ChannelT], float]
"""Callback function that returns the edge cost of a channel."""


def make_csr(
    nodes: list[NodeT],
    channels: list[ChannelT],
    metric_func: MetricFunc,
) -> csr_matrix:
    """
    Build a symmetric weighted adjacency matrix for SciPy CSR.

    Args:
        nodes: a list of quantum nodes or classic nodes
        channels: a list of quantum channels or classic channels
        metric_func: Function mapping a channel -> float weight.

    Returns:
        csr_matrix: (n x n) weighted adjacency matrix.
    """
    n = len(nodes)
    node_index = {nd: i for i, nd in enumerate(nodes)}

    rows, cols, data = [], [], []
    for ch in channels:
        if len(ch.node_list) != 2:
            raise NetworkRouteError("broken link")
        a, b = ch.node_list
        ai, bi = node_index[a], node_index[b]
        w = float(metric_func(ch))
        # undirected: add both directions
        rows.extend([ai, bi])
        cols.extend([bi, ai])
        data.extend([w, w])

    return csr_matrix(
        (np.asarray(data, np.float64), (np.asarray(rows, np.int32), np.asarray(cols, np.int32))),
        shape=(n, n),
    )


class NetworkRouteError(Exception):
    pass


class RouteImpl(Generic[NodeT, ChannelT]):
    """This is the route protocol interface"""

    def __init__(self, name: str = "route") -> None:
        self.name = name

    def build(self, nodes: list[NodeT], channels: list[ChannelT]) -> None:
        """Build static route tables for each nodes

        Args:
            nodes: a list of quantum nodes or classic nodes
            channels: a list of quantum channels or classic channels

        """
        raise NotImplementedError

    def query(self, src: NodeT, dest: NodeT) -> list[tuple[float, NodeT, list[NodeT]]]:
        """Query the metric, nexthop and the path

        Args:
            src: the source node
            dest: the destination node

        Returns:
            A list of route paths. The result should be sorted by priority.
            The element is a tuple containing: metric, the next-hop and the whole path.

        """
        raise NotImplementedError
