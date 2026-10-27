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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Final, override

from mqns.entity.node.app import Application
from mqns.entity.node.node import Node
from mqns.simulator import Simulator

if TYPE_CHECKING:
    from mqns.entity.memory import QuantumMemory
    from mqns.entity.operator import QuantumOperator
    from mqns.entity.qchannel import QuantumChannel


class QNode(Node):
    """QNode is a quantum node in the quantum network. Inherits Node and add quantum elements."""

    _memory: "QuantumMemory|None" = None

    operators: Final[Sequence["QuantumOperator"]]
    """Quantum operators on this node."""

    qchannels: Final[Sequence["QuantumChannel"]]
    """Quantum channels connected to this node."""

    def __init__(self, name: str, *, apps: Iterable[Application] | None = None, memory: "QuantumMemory|None" = None):
        """
        Args:
            name: Node name.
            apps: Applications on the node.
                  More applications can be added via :py:meth:`~mqns.entity.node.Node.add_apps`.
                  The caller is responsible for ``deepcopy`` if needed, so that each node has a separate instance.
            memory: Quantum memory to assign to this node.
                    It can also be assigned later via :py:attr:`memory` setter.
        """
        super().__init__(name, apps=apps)
        self._operators: list[QuantumOperator] = []
        self._qchannels: list[QuantumChannel] = []
        self._qchannel_by_dst: dict[Node, QuantumChannel] = {}

        self.operators = self._operators
        self.qchannels = self._qchannels

        if memory is not None:
            self.memory = memory

    @override
    def _node_install(self) -> None:
        self._install_channels(self._qchannels, self._qchannel_by_dst)

        if self._memory:
            self._memory.install(self.simulator)
        for operator in self.operators:
            operator.install(self.simulator)

    @property
    def memory(self) -> "QuantumMemory":
        """
        Retrieve or assign associated QuantumMemory.

        Raises:
            LookupError: Memory does not exist.

        The setter is available prior to calling :py:meth:`~mqns.entity.node.Node.install`.
        """
        if self._memory is None:
            raise LookupError(f"{self} does not have memory")
        return self._memory

    @memory.setter
    def memory(self, value: "QuantumMemory"):
        Simulator.ensure_not_installed_to(self)
        value.node = self
        self._memory = value

    def add_operator(self, operator: "QuantumOperator") -> None:
        """
        Add a quantum operator in this node.

        This function is available prior to calling :py:meth:`~mqns.entity.node.Node.install`.
        """
        Simulator.ensure_not_installed_to(self)
        operator.set_own(self)
        self._operators.append(operator)

    def add_qchannel(self, qchannel: "QuantumChannel") -> None:
        """
        Add a quantum channel in this node.

        This function is available prior to calling :py:meth:`~mqns.entity.node.Node.install`.
        """
        self._add_channel(qchannel, self._qchannels)

    def get_qchannel(self, dst: "QNode") -> "QuantumChannel":
        """
        Retrieve the quantum channel that connects to ``dst``.

        Raises:
            LookupError: channel does not exist
        """
        return self._get_channel(dst, self._qchannel_by_dst)

    def __repr__(self) -> str:
        return f"<qnode {self.name}>"
