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

from qns.network.topology.gridtopo import GridTopology
from qns.network.topology.topo import TopologyInitKwargs

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


class LinearTopology(GridTopology):
    """
    LinearTopology creates a linear topology with `nodes_number` nodes.
    """

    def __init__(self, nodes_number: int, **kwargs: Unpack[TopologyInitKwargs]):
        super().__init__((1, nodes_number), **kwargs)
        # A linear topology is a special case of a grid topology with one row and nodes_number columns.
