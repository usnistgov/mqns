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


from typing import TYPE_CHECKING, Any

from typing_extensions import override

from mqns.entity.memory.memory_qubit import MemoryQubit
from mqns.entity.node import QNode
from mqns.models.core import QuantumModel
from mqns.simulator import Event, Time

if TYPE_CHECKING:
    from mqns.entity.memory.memory import QuantumMemory


class MemoryReadRequestEvent(Event):
    """``MemoryReadRequestEvent`` is the event that request a memory read"""

    def __init__(
        self,
        memory: "QuantumMemory",
        key: QuantumModel | str,
        *,
        t: Time,
        name: str | None = None,
        by: Any = None,
    ):
        super().__init__(t=t, name=name, by=by)
        self.memory = memory
        self.key = key

    @override
    def invoke(self) -> None:
        self.memory.handle(self)


class MemoryReadResponseEvent(Event):
    """``MemoryReadResponseEvent`` is the event that returns the memory read result"""

    def __init__(
        self,
        node: QNode,
        result: tuple[MemoryQubit, QuantumModel | None] | None,
        *,
        request: MemoryReadRequestEvent,
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


class MemoryWriteRequestEvent(Event):
    """``MemoryWriteRequestEvent`` is the event that request a memory write"""

    def __init__(self, memory: "QuantumMemory", qubit: QuantumModel, *, t: Time, name: str | None = None, by: Any = None):
        super().__init__(t=t, name=name, by=by)
        self.memory = memory
        self.qubit = qubit

    @override
    def invoke(self) -> None:
        self.memory.handle(self)


class MemoryWriteResponseEvent(Event):
    """``MemoryWriteResponseEvent`` is the event that returns the memory write result"""

    def __init__(
        self,
        node: QNode,
        result: MemoryQubit | None = None,
        *,
        request: MemoryWriteRequestEvent,
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
