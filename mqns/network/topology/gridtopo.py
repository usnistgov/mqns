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

import math
from typing import Unpack, override

import numpy as np

from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.topology.topo import Topology, TopologyInitKwargs


class GridTopology(Topology):
    """
    GridTopology builds a grid topology with a rectangle shape.
    If `shape` is a tuple, it specifies how many rows and columns are in the topology.
    If `shape` is an integer, it specifies total number of nodes and must be a perfect square number.
    """

    def __init__(self, shape: int | tuple[int, int], **kwargs: Unpack[TopologyInitKwargs]):
        super().__init__(int(np.prod(shape)), **kwargs)
        if isinstance(shape, tuple):
            self.rows, self.cols = shape
        else:
            size = int(math.sqrt(shape))
            self.rows, self.cols = size, size
        assert self.rows * self.cols == self.nodes_number

    @override
    def build(self) -> tuple[list[QNode], list[QuantumChannel]]:
        nl: list[QNode] = []
        ll: list[QuantumChannel] = []

        for i in range(self.nodes_number):
            n = QNode(f"n{i + 1}")
            nl.append(n)

        def qc(a: int, b: int):
            link = QuantumChannel(name=f"l{a},{b}", **self.qchannel_args)
            ll.append(link)
            nl[a].add_qchannel(link)
            nl[b].add_qchannel(link)

        # horizontal links
        for r in range(self.rows):
            idx = r * self.cols
            for c in range(self.cols - 1):
                qc(idx + c, idx + c + 1)

        # vertical links
        for r in range(self.rows - 1):
            idx = r * self.cols
            for c in range(self.cols):
                qc(idx + c, idx + c + self.cols)

        self._add_apps(nl)
        self._add_memories(nl)
        return nl, ll
