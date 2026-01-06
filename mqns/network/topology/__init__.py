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

from mqns.network.topology.basictopo import BasicTopology
from mqns.network.topology.customtopo import CustomTopology
from mqns.network.topology.gridtopo import GridTopology
from mqns.network.topology.lineartopo import LinearTopology
from mqns.network.topology.randomtopo import RandomTopology
from mqns.network.topology.topo import ClassicTopology, Topology, TopologyInitKwargs
from mqns.network.topology.treetopo import TreeTopology
from mqns.network.topology.waxmantopo import WaxmanTopology

__all__ = [
    "BasicTopology",
    "ClassicTopology",
    "CustomTopology",
    "GridTopology",
    "LinearTopology",
    "RandomTopology",
    "Topology",
    "TopologyInitKwargs",
    "TreeTopology",
    "WaxmanTopology",
]
