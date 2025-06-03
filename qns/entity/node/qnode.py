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

from typing import TYPE_CHECKING

from qns.entity.node.app import Application
from qns.entity.node.node import Node
from qns.simulator import Simulator

if TYPE_CHECKING:
    from qns.entity import QuantumChannel, QuantumMemory, QuantumOperator

class QNode(Node):
    """QNode is a quantum node in the quantum network. Inherits Node and add quantum elements.
    """

    def __init__(self, name: str, *, apps: list[Application]|None = None):
        """Args:
        name (str): the node's name
        apps (List[Application]): the installing applications.

        """
        super().__init__(name=name, apps=apps)
        self.qchannels: list["QuantumChannel"] = []
        self.memory: "QuantumMemory|None" = None
        self.operators: list["QuantumOperator"] = []
        self.qroute_table = [] # XXX unused

    def install(self, simulator: Simulator) -> None:
        super().install(simulator)
        # initiate sub-entities
        from qns.entity import QuantumChannel, QuantumMemory, QuantumOperator
        if self.memory is not None:
            assert isinstance(self.memory, QuantumMemory)
            self.memory.install(simulator)
        for qchannel in self.qchannels:
            assert isinstance(qchannel, QuantumChannel)
            qchannel.install(simulator)
        for operator in self.operators:
            assert isinstance(operator, QuantumOperator)
            operator.install(simulator)

    def set_memory(self, memory: "QuantumMemory"):
        """Add a quantum memory in this QNode

        Args:
            memory (Memory): the quantum memory

        This function is available prior to calling .install().
        """
        assert self._simulator is None
        memory.node = self
        self.memory = memory

    def get_memory(self) -> "QuantumMemory":
        """Get the memory

        Raises:
            IndexError - memory does not exist
        """
        if self.memory is None:
            raise IndexError(f"node {repr(self)} does not have memory")
        return self.memory

    def add_operator(self, operator: "QuantumOperator"):
        """Add a quantum operator in this node

        Args:
            operator (QuantumOperator): the quantum operator

        This function is available prior to calling .install().
        """
        assert self._simulator is None
        operator.set_own(self)
        self.operators.append(operator)

    def add_qchannel(self, qchannel: "QuantumChannel"):
        """Add a quantum channel in this QNode

        Args:
            qchannel (QuantumChannel): the quantum channel

        This function is available prior to calling .install().
        """
        assert self._simulator is None
        qchannel.node_list.append(self)
        self.qchannels.append(qchannel)

    def get_qchannel(self, dst: "QNode") -> "QuantumChannel":
        """Get the quantum channel that connects to the `dst`

        Args:
            dst (QNode): the destination

        Raises:
            IndexError - channel does not exist
        """
        for qchannel in self.qchannels:
            if dst in qchannel.node_list and self in qchannel.node_list:
                return qchannel
        raise IndexError(f"qchannel from {repr(self)} to {repr(dst)} does not exist")

    def __repr__(self) -> str:
        if self.name is not None:
            return f"<qnode {self.name}>"
        return super().__repr__()
