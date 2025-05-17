#    Dynamic-QNetSim: a simulator for comparative evaluation of quantum routing strategies
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

from qns.entity.node.app import Application
from qns.entity.node.node import Node
from qns.simulator import Simulator


class QNode(Node):
    """QNode is a quantum node in the quantum network. Inherits Node and add quantum elements.
    """

    def __init__(self, name: str = None, apps: list[Application] = None):
        """Args:
        name (str): the node's name
        apps (List[Application]): the installing applications.

        """
        super().__init__(name=name, apps=apps)
        self.qchannels = []
        self.memory = None
        self.operators = []
        self.qroute_table = []

    def install(self, simulator: Simulator) -> None:
        super().install(simulator)
        # initiate sub-entities

        from qns.entity import QuantumMemory
        assert (isinstance(self.memory, QuantumMemory))
        self.memory.install(simulator)

        for qchannel in self.qchannels:
            from qns.entity import QuantumChannel
            assert (isinstance(qchannel, QuantumChannel))
            qchannel.install(simulator)
        for operator in self.operators:
            from qns.entity import QuantumOperator
            assert (isinstance(operator, QuantumOperator))
            operator.install(simulator)

    def set_memory(self, memory):
        """Add a quantum memory in this QNode

        Args:
            memory (Memory): the quantum memory

        """
        memory.node = self
        self.memory = memory

    def get_memory(self):
        """Get the memory
        """
        return self.memory

    def add_operator(self, operator):
        """Add a quantum operator in this node

        Args:
            operator (QuantumOperator): the quantum operator

        """
        operator.set_own(self)
        self.operators.append(operator)

    def add_qchannel(self, qchannel):
        """Add a quantum channel in this QNode

        Args:
            qchannel (QuantumChannel): the quantum channel

        """
        qchannel.node_list.append(self)
        self.qchannels.append(qchannel)

    def get_qchannel(self, dst: "QNode"):
        """Get the quantum channel that connects to the `dst`

        Args:
            dst (QNode): the destination

        """
        for qchannel in self.qchannels:
            if dst in qchannel.node_list and self in qchannel.node_list:
                return qchannel
        return None

    def __repr__(self) -> str:
        if self.name is not None:
            return f"<qnode {self.name}>"
        return super().__repr__()
