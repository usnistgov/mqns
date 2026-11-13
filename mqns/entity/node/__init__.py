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

from typing import cast

from mqns.entity.node.app import Application
from mqns.entity.node.controller import Controller
from mqns.entity.node.node import Node
from mqns.entity.node.qnode import QNode

type NodePair = str | tuple[str, str]
"""
Two node names on a channel or routing path.

It could be written as either:

* A tuple of two node names, such as ``("A", "B")``.
* A string delimited by hyphen, such as ``"A-B"``.
"""


def split_node_pair(np: NodePair) -> tuple[str, str]:
    """
    Convert :py:class:`NodePair` to a tuple of two node names.
    """
    if isinstance(np, str):
        tokens = np.split("-")
        if len(tokens) != 2:
            raise ValueError(f"expect two node names in '{np}'")
        return cast(tuple[str, str], tuple(tokens))
    return np


__all__ = [
    "Application",
    "Controller",
    "Node",
    "NodePair",
    "QNode",
    "split_node_pair",
]

for name in __all__:
    if name in ("NodePair",):
        continue
    globals()[name].__module__ = __name__
