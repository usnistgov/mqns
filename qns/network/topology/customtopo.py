#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
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


from typing import TypedDict

from qns.entity.cchannel import ClassicChannel, ClassicChannelInitKwargs
from qns.entity.memory import QuantumMemory, QuantumMemoryInitKwargs
from qns.entity.node import Application, Controller, QNode
from qns.entity.qchannel import QuantumChannel, QuantumChannelInitKwargs
from qns.network.topology.topo import Topology

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


class TopoQNode(TypedDict):
    name: str
    memory: QuantumMemoryInitKwargs
    apps: list[Application]


class TopoQChannel(TypedDict):
    node1: str
    """first node name"""
    node2: str
    """second node name"""
    capacity1: NotRequired[int]
    """quantity of memory qubits assigned to this qchannel at node1, defaults to `capacity`"""
    capacity2: NotRequired[int]
    """quantity of memory qubits assigned to this qchannel at node2, defaults to `capacity`"""
    capacity: NotRequired[int]
    """quantity of memory qubits assigned to this qchannel at each node, defaults to 1"""
    parameters: QuantumChannelInitKwargs
    """qchannel constructor arguments"""


class TopoCChannel(TypedDict):
    node1: str
    node2: str
    parameters: ClassicChannelInitKwargs


class TopoController(TypedDict):
    name: str
    apps: list[Application]


class Topo(TypedDict):
    qnodes: list[TopoQNode]
    qchannels: list[TopoQChannel]
    cchannels: list[TopoCChannel]
    controller: NotRequired[TopoController]


class CustomTopology(Topology):
    """
    CustomTopology builds a topology from a JSON-like dict structure.

    Nodes and channels are individually specified and can have heterogeneous parameters.
    """

    def __init__(self, topo: Topo):
        super().__init__(0)
        self.topo = topo

    def build(self) -> tuple[list[QNode], list[QuantumChannel]]:
        qnl: list[QNode] = []
        qcl: list[QuantumChannel] = []

        # Create quantum nodes
        for node in self.topo["qnodes"]:
            qn = QNode(node["name"])
            for app in node["apps"]:
                qn.add_apps(app)

            # Assign a new memory
            memory_args = node["memory"]
            m = QuantumMemory(name=qn.name, node=qn, **memory_args)
            qn.set_memory(m)

            qnl.append(qn)

        # Create quantum channels and assign memories with proper capacity
        for ch in self.topo["qchannels"]:
            node1, node2 = ch["node1"], ch["node2"]
            link = QuantumChannel(name=f"q_{node1},{node2}", **ch["parameters"])
            qcl.append(link)

            # Attach quantum channel to nodes
            for qn in qnl:
                if qn.name in (node1, node2):
                    qn.add_qchannel(link)

            link.assign_memory_qubits(
                capacity={
                    node1: ch.get("capacity1", ch.get("capacity", 1)),
                    node2: ch.get("capacity2", ch.get("capacity", 1)),
                }
            )

        if "controller" in self.topo:
            self.controller = Controller(name=self.topo["controller"]["name"], apps=self.topo["controller"]["apps"])

        self.qnl = qnl
        return qnl, qcl

    def add_cchannels(self, **kwargs):
        if len(kwargs) != 0:
            raise TypeError("CustomTopology.add_cchannels() does not accept classic_topo= keyword")

        ccl: list[ClassicChannel] = []
        for ch in self.topo["cchannels"]:
            link = ClassicChannel(name=f"c_{ch['node1']},{ch['node2']}", **ch["parameters"])
            ccl.append(link)

            for node in self.qnl + [self.controller]:
                if node and (node.name == ch["node1"] or node.name == ch["node2"]):
                    # Attach classic channel to nodes
                    node.add_cchannel(link)

        return ccl
