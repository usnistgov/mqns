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


from collections.abc import Iterable
from copy import deepcopy
from typing import Any, TypedDict, cast

from typing_extensions import NotRequired, override

from mqns.entity.cchannel import ClassicChannel, ClassicChannelInitKwargs
from mqns.entity.memory import QuantumMemory, QuantumMemoryInitKwargs
from mqns.entity.node import Application, Controller, Node, QNode
from mqns.entity.qchannel import QuantumChannel, QuantumChannelInitKwargs
from mqns.network.topology.topo import ClassicTopology, Topology


class TopoQNode(TypedDict):
    name: str
    """Node name."""
    memory: NotRequired[QuantumMemoryInitKwargs]
    """
    Memory parameters.

    If omitted, use `memory_args` passed to CustomTopo constructor.
    """
    apps: NotRequired[list[Application]]
    """
    Applications installed on the node.

    If omitted, use `nodes_apps` passed to CustomTopo constructor.
    """


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
    """
    List of quantum nodes.
    """
    qchannels: list[TopoQChannel]
    """
    List of quantum channels.
    """
    cchannels: NotRequired[list[TopoCChannel]]
    """
    List of classic channels.

    If omitted, the topology must be used with ClassicTopology.Follow and to reuse compatible quantum channel parameters.
    If specified, the topology must be used with ClassicTopology.Empty (default) to use these specifications.
    """
    controller: NotRequired[TopoController]
    """
    Controller parameters.
    """


def _qchannel_to_cchannel(qc: TopoQChannel) -> TopoCChannel:
    parameters = {}
    for key, value in qc["parameters"].items():
        if key in ClassicChannelInitKwargs.__annotations__.keys():
            parameters[key] = value
    return {"node1": qc["node1"], "node2": qc["node2"], "parameters": cast(Any, parameters)}


class CustomTopology(Topology):
    """
    CustomTopology builds a topology from a JSON-like dict structure.

    Nodes and channels are individually specified and can have heterogeneous parameters.
    """

    def __init__(
        self,
        topo: Topo,
        *,
        nodes_apps: list[Application] = [],
        memory_args: QuantumMemoryInitKwargs = {},
    ):
        super().__init__(len(topo["qnodes"]), nodes_apps=nodes_apps, memory_args=memory_args)
        self.topo = topo
        self._node_by_name = dict[str, Node]()

    def _save_node(self, node: Node):
        assert node.name not in self._node_by_name, f"duplicate node name {node.name}"
        self._node_by_name[node.name] = node

    @override
    def build(self) -> tuple[list[QNode], list[QuantumChannel]]:
        qnl: list[QNode] = []
        qcl: list[QuantumChannel] = []

        # Create quantum nodes
        for node in self.topo["qnodes"]:
            qn = QNode(node["name"])
            qn.add_apps(node["apps"] if "apps" in node else deepcopy(self.nodes_apps))

            # Assign a new memory
            memory_args = node.get("memory", self.memory_args)
            m = QuantumMemory(f"{qn.name}.memory", **memory_args)
            qn.set_memory(m)

            qnl.append(qn)
            self._save_node(qn)

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
            ctrl = self.topo["controller"]
            self.controller = Controller(name=ctrl["name"], apps=ctrl["apps"])
            self._save_node(self.controller)

        return qnl, qcl

    @override
    def add_cchannels(self, *, classic_topo: ClassicTopology = ClassicTopology.Empty, **_):
        if classic_topo == ClassicTopology.Follow:
            assert "cchannels" not in self.topo
            return self._add_cchannels_from([_qchannel_to_cchannel(qc) for qc in self.topo["qchannels"]])
        else:
            assert classic_topo == ClassicTopology.Empty
            assert "cchannels" in self.topo
            return self._add_cchannels_from(self.topo["cchannels"])

    def _add_cchannels_from(self, cchannels: Iterable[TopoCChannel]):
        ccl: list[ClassicChannel] = []
        for ch in cchannels:
            link = ClassicChannel(name=f"c_{ch['node1']},{ch['node2']}", **ch["parameters"])
            ccl.append(link)
            self._node_by_name[ch["node1"]].add_cchannel(link)
            self._node_by_name[ch["node2"]].add_cchannel(link)
        return ccl
