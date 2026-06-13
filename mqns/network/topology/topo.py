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

import copy
import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Literal, TypedDict, Unpack

from mqns.entity.cchannel import ClassicChannel, ClassicChannelInitKwargs
from mqns.entity.memory import QuantumMemory, QuantumMemoryInitKwargs
from mqns.entity.node import Application, Controller, Node, QNode
from mqns.entity.qchannel import QuantumChannel, QuantumChannelInitKwargs


class TopologyInitKwargs(TypedDict, total=False):
    nodes_naming: Literal["n1", "A"]
    nodes_apps: list[Application]
    qchannel_args: QuantumChannelInitKwargs
    cchannel_args: ClassicChannelInitKwargs
    memory_args: QuantumMemoryInitKwargs


class ClassicTopology(Enum):
    """Indicates how to derive classical topology from quantum topology."""

    Empty = 1
    """No connection."""
    All = 2
    """All pairs -- every node connects to every other node."""
    Follow = 3
    """Follow the same topology as quantum channels."""


class Topology(ABC):
    """
    Topology is a factory for quantum and classic network topology used to build QuantumNetwork.
    """

    def __init__(self, nodes_number: int, **kwargs: Unpack[TopologyInitKwargs]):
        """
        Args:
            nodes_number: Total number of quantum nodes.
            nodes_naming: Naming convention for the nodes.
            nodes_apps: Applications installed on all nodes.
            qchannel_args: Default quantum channel arguments.
            cchannel_args: Default channel channel arguments.
            memory_args: Default quantum memory arguments.
        """
        self.nodes_number = nodes_number
        self.nodes_naming = kwargs.get("nodes_naming", "n1")
        self.nodes_apps = kwargs.get("nodes_apps", [])
        self.qchannel_args = kwargs.get("qchannel_args", {})
        self.cchannel_args = kwargs.get("cchannel_args", {})
        self.memory_args = kwargs.get("memory_args", {})
        self.controller: Controller | None = None

    @abstractmethod
    def build(self) -> tuple[list[QNode], list[QuantumChannel]]:
        """
        Build the topology.

        Returns: list of nodes and quantum channels.
        """

    def _name_node(self, i: int) -> str:
        if self.nodes_naming == "n1":
            return f"n{1 + i}"
        elif self.nodes_naming == "A":
            if i > 26:
                raise ValueError("too many nodes for nodes_naming='A'")
            return chr(0x41 + i)
        raise ValueError("unknown nodes_naming")

    def _name_channel(self, a: int, b: int) -> str:
        if self.nodes_naming == "n1":
            return f"l{1 + a},{1 + b}"
        elif self.nodes_naming == "A":
            return f"{chr(0x41 + a)}-{chr(0x41 + b)}"
        raise ValueError("unknown nodes_naming")

    def _add_apps(self, nl: Iterable[QNode]) -> None:
        """
        Add apps for all nodes in ``nl``.

        Args:
            nl: List of quantum nodes.
        """
        for n in nl:
            n.add_apps(copy.deepcopy(self.nodes_apps))

    def _add_memories(self, nl: Iterable[QNode]) -> None:
        """
        Add quantum memories to all nodes in ``nl``.

        Args:
            nl: List of quantum nodes.
        """
        for node in nl:
            node.memory = QuantumMemory(node.name, **self.memory_args)

    def add_cchannels(
        self,
        *,
        classic_topo: ClassicTopology = ClassicTopology.Empty,
        nl: Iterable[QNode] = [],
        ll: Iterable[QuantumChannel] = [],
    ) -> list[ClassicChannel]:
        """
        Build classic network topology.

        Args:
            classic_topo: Classic topology build strategy.
            nl: List of quantum nodes.
            ll: List of quantum channels.
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

    def connect_controller(self, nl: Iterable[Node], **kwargs: Unpack[ClassicChannelInitKwargs]) -> list[ClassicChannel]:
        """
        Create a cchannel from the controller to each node.

        Args:
            nl: list of non-controller nodes.

        Returns:
            List of classical channels.

        Raises:
            RuntimeError: controller does not exist.

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
        except AttributeError:  # controller is not part of a network
            pass

        return cchannels
