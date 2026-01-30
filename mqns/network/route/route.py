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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from scipy.sparse import csr_matrix

from mqns.entity.base_channel import BaseChannel
from mqns.entity.node import Node

type MetricFunc[C: BaseChannel] = Callable[[C], float]
"""Callback function that returns the edge cost of a channel."""


def make_csr[N: Node, C: BaseChannel](
    nodes: list[N],
    channels: list[C],
    metric_func: MetricFunc[C],
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

    rows = np.zeros((2 * len(channels),), dtype=np.int32)
    cols = np.zeros((2 * len(channels),), dtype=np.int32)
    data = np.zeros((2 * len(channels),), dtype=np.float64)
    for i, ch in enumerate(channels):
        assert len(ch.node_list) == 2
        a, b = ch.node_list
        ai, bi = node_index[a], node_index[b]
        w = float(metric_func(ch))
        # undirected: add both directions
        rows[2 * i + 0], cols[2 * i + 0], data[2 * i + 0] = ai, bi, w
        rows[2 * i + 1], cols[2 * i + 1], data[2 * i + 1] = bi, ai, w

    return csr_matrix((data, (rows, cols)), shape=(n, n))


class RouteQueryResult[N: Node](NamedTuple):
    metric: float
    next_hop: N
    route: list[N]


class RouteAlgorithm[N: Node, C: BaseChannel](ABC):
    """
    Represents a routing algorithm that computes routes between two nodes in a network.
    """

    def __init__(self, name: str, metric_func: MetricFunc[C] | None = None) -> None:
        self.name = name

        if metric_func is None:
            self.metric_func: MetricFunc[C] = lambda _: 1  # hop count
            self.unweighted = True
        else:
            self.metric_func = metric_func
            self.unweighted = False

    @abstractmethod
    def build(self, nodes: list[N], channels: list[C]) -> None:
        """
        Build static route tables.

        Args:
            nodes: a list of quantum nodes or classic nodes.
            channels: a list of quantum channels or classic channels.
        """
        pass

    @abstractmethod
    def query(self, src: N, dst: N) -> list[RouteQueryResult[N]]:
        """
        Query the metric, next-hop and the path.

        Args:
            src: the source node.
            dst: the destination node.

        Returns:
            A list of route paths. The result should be sorted by priority.
        """
        pass
