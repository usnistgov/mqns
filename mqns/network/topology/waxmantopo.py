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

import itertools
from copy import deepcopy

import numpy as np
from typing_extensions import Unpack, override

from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.topology.topo import Topology, TopologyInitKwargs
from mqns.utils.rnd import get_rand


class WaxmanTopology(Topology):
    """WaxmanTopology is the random topology generator using Waxman's model."""

    def __init__(self, nodes_number: int, size: float, alpha: float, beta: float, **kwargs: Unpack[TopologyInitKwargs]):
        """Args:
        nodes_number (int): the number of Qnodes
        size (float): the area size (meter)
        alpha (float): alpha parameter in Waxman's model
        beta (float): beta parameter in Waxman's model

        """
        super().__init__(nodes_number, **kwargs)
        self.size = size
        self.alpha = alpha
        self.beta = beta

    @override
    def build(self) -> tuple[list[QNode], list[QuantumChannel]]:
        nl: list[QNode] = []
        ll: list[QuantumChannel] = []

        location_table: dict[QNode, tuple[float, float]] = {}
        distance_table: dict[tuple[QNode, QNode], float] = {}

        for i in range(self.nodes_number):
            n = QNode(f"n{i + 1}")
            nl.append(n)
            x = get_rand() * self.size
            y = get_rand() * self.size
            location_table[n] = (x, y)

        L = 0
        cb = list(itertools.combinations(nl, 2))
        for n1, n2 in cb:
            tmp_l = np.sqrt(
                (location_table[n1][0] - location_table[n2][0]) ** 2 + (location_table[n1][1] - location_table[n2][1]) ** 2
            )
            distance_table[(n1, n2)] = tmp_l
            L = max(L, tmp_l)

        for n1, n2 in cb:
            if n1 == n2:
                continue
            d = distance_table[(n1, n2)]
            p = self.alpha * np.exp(-d / (self.beta * L))
            if get_rand() < p:
                qchannel_args = deepcopy(self.qchannel_args)
                qchannel_args.setdefault("length", d)
                link = QuantumChannel(name=f"l{n1}-{n2}", **qchannel_args)
                ll.append(link)
                n1.add_qchannel(link)
                n2.add_qchannel(link)
        self._add_apps(nl)
        self._add_memories(nl)
        return nl, ll
