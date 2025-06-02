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

from typing import Generic, TypeVar

from qns.entity.cchannel import ClassicChannel
from qns.entity.node import Node, QNode
from qns.entity.qchannel import QuantumChannel

NodeT = TypeVar("NodeT", bound=Node|QNode)
ChannelT = TypeVar("ChannelT", bound=ClassicChannel|QuantumChannel)


class NetworkRouteError(Exception):
    pass


class RouteImpl(Generic[NodeT, ChannelT]):
    """This is the route protocol interface
    """

    def __init__(self, name: str = "route") -> None:
        self.name = name

    def build(self, nodes: list[NodeT], channels: list[ChannelT]) -> None:
        """Build static route tables for each nodes

        Args:
            channels: a list of quantum channels or classic channels

        """
        raise NotImplementedError

    def query(self, src: NodeT, dest: NodeT) -> list[tuple[float, NodeT, list[NodeT]]]:
        """Query the metric, nexthop and the path

        Args:
            src: the source node
            dest: the destination node

        Returns:
            A list of route paths. The result should be sorted by the priority.
            The element is a tuple containing: metric, the next-hop and the whole path.

        """
        raise NotImplementedError
