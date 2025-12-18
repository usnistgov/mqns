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

import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import TypedDict, Unpack

from mqns.entity.cchannel import ClassicChannel, ClassicChannelInitKwargs
from mqns.entity.memory import QuantumMemory, QuantumMemoryInitKwargs
from mqns.entity.node import Application, Controller, NodeT, QNode
from mqns.entity.qchannel import QuantumChannel, QuantumChannelInitKwargs


class TopologyInitKwargs(TypedDict, total=False):
    nodes_apps: list[Application]
    qchannel_args: QuantumChannelInitKwargs
    cchannel_args: ClassicChannelInitKwargs
    memory_args: QuantumMemoryInitKwargs


class ClassicTopology(Enum):
    Empty = 1
    All = 2
    Follow = 3


class Topology(ABC):
    """Topology is a factory for QuantumNetwork"""

    def __init__(self, nodes_number: int, **kwargs: Unpack[TopologyInitKwargs]):
        """Args:
        nodes_number: the number of Qnodes
        nodes_apps: apps will be installed to all nodes
        qchannel_args: default quantum channel arguments
        cchannel_args: default channel channel arguments
        memory_args: default quantum memory arguments

        """
        self.nodes_number = nodes_number
        self.nodes_apps = kwargs.get("nodes_apps", [])
        self.qchannel_args = kwargs.get("qchannel_args", {})
        self.cchannel_args = kwargs.get("cchannel_args", {})
        self.memory_args = kwargs.get("memory_args", {})
        self.controller: Controller | None = None

    @abstractmethod
    def build(self) -> tuple[list[QNode], list[QuantumChannel]]:
        """Build the special topology

        Returns:
            the list of QNodes and the list of QuantumChannel

        """
        pass

    def _add_apps(self, nl: list[QNode]):
        """Add apps for all nodes in `nl`

        Args:
            nl (List[QNode]): a list of quantum nodes

        """
        for n in nl:
            n.add_apps(deepcopy(self.nodes_apps))

    def _add_memories(self, nl: list[QNode]):
        """Add quantum memories to all nodes in `nl`

        Args:
            nl (List[QNode]): a list of quantum nodes

        """
        for node in nl:
            node.memory = QuantumMemory(f"{node.name}.memory", **self.memory_args)

    def add_cchannels(
        self, *, classic_topo: ClassicTopology = ClassicTopology.Empty, nl: list[QNode] = [], ll: list[QuantumChannel] = []
    ) -> list[ClassicChannel]:
        """Build classic network topology

        Args:
            classic_topo (ClassicTopology): Classic topology,
                ClassicTopology.Empty -> no connection
                ClassicTopology.All -> every nodes are connected directly
                ClassicTopology.Follow -> follow the quantum topology
            nl (List[mqns.entity.node.node.QNode]): a list of quantum nodes
            ll (List[mqns.entity.qchannel.qchannel.QuantumChannel]): a list of quantum channels

        """
        cchannel_list: list[ClassicChannel] = []
        if classic_topo == ClassicTopology.All:
            topo = list(itertools.combinations(nl, 2))
            for idx, (src, dst) in enumerate(topo):
                cchannel = ClassicChannel(f"c{idx + 1}", **self.cchannel_args)
                src.add_cchannel(cchannel=cchannel)
                dst.add_cchannel(cchannel=cchannel)
                cchannel_list.append(cchannel)
        elif classic_topo == ClassicTopology.Follow:
            for idx, qchannel in enumerate(ll):
                node_list = qchannel.node_list
                cchannel = ClassicChannel(f"c-{qchannel.name}", **self.cchannel_args)
                for n in node_list:
                    n.add_cchannel(cchannel=cchannel)
                cchannel_list.append(cchannel)

        return cchannel_list

    def connect_controller(self, nl: list[NodeT], **kwargs: Unpack[ClassicChannelInitKwargs]) -> list[ClassicChannel]:
        """
        Create a cchannel from the controller to each node.

        Args:
            nl: list of non-controller nodes.

        Returns:
            List of classical channels.

        Raises:
            RuntimeError - controller does not exist.

        Notes:
            If the controller is part of a network, newly created cchannels are automatically added to the network.
        """
        if self.controller is None:
            raise RuntimeError("controller does not exist")

        cchannels: list[ClassicChannel] = []
        for node in nl:
            cchannel = ClassicChannel(f"ctrl-{node.name}", **kwargs)
            self.controller.add_cchannel(cchannel)
            node.add_cchannel(cchannel)
            cchannels.append(cchannel)

        try:
            net = self.controller.network
            for cchannel in cchannels:
                net.add_cchannel(cchannel)
        except IndexError:
            pass

        return cchannels
