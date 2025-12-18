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

from typing import TYPE_CHECKING, Any, final, override

from mqns.entity.node.qnode import QNode
from mqns.models.core import QuantumModel
from mqns.simulator import Event, Time

if TYPE_CHECKING:
    from mqns.entity.operator.operator import QuantumOperator


@final
class OperateRequestEvent(Event):
    """``OperateRequestEvent`` is the event that request a operator to handle"""

    def __init__(
        self,
        operator: "QuantumOperator",
        qubits: list[QuantumModel] = [],
        *,
        t: Time,
        name: str | None = None,
        by: Any = None,
    ):
        super().__init__(t=t, name=name, by=by)
        self.operator = operator
        self.qubits = qubits

    @override
    def invoke(self) -> None:
        self.operator.handle(self)


@final
class OperateResponseEvent(Event):
    """``OperateResponseEvent`` is the event that returns the operating result"""

    def __init__(
        self,
        node: QNode,
        result: int | list[int] | None = None,
        *,
        request: OperateRequestEvent,
        t: Time,
        name: str | None = None,
        by: Any = None,
    ):
        super().__init__(t=t, name=name, by=by)
        self.node = node
        self.result = result
        self.request = request

    @override
    def invoke(self) -> None:
        self.node.handle(self)
