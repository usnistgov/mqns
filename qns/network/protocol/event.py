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

from enum import Enum, auto
from typing import Any

from qns.entity.memory import MemoryQubit
from qns.entity.node import QNode
from qns.simulator.event import Event
from qns.simulator.ts import Time


class TypeEnum(Enum):
    ADD = auto()
    REMOVE = auto()


class ManageActiveChannels(Event):
    """
    Event sent by Forwarder to request LinkLayer to start generating EPRs over a qchannel.
    """

    def __init__(
        self,
        node: QNode,
        neighbor: QNode,
        type: TypeEnum,
        path_id: int | None = None,
        *,
        t: Time,
        name: str | None = None,
        by: Any = None,
    ):
        super().__init__(t=t, name=name, by=by)
        self.node = node
        self.neighbor = neighbor
        self.path_id = path_id
        self.type = type

    def invoke(self) -> None:
        self.node.handle(self)


class QubitDecoheredEvent(Event):
    """
    Event sent by Memory to inform LinkLayer about a decohered qubit.
    """

    def __init__(self, node: QNode, qubit: MemoryQubit, *, t: Time, name: str | None = None, by: Any = None):
        super().__init__(t=t, name=name, by=by)
        self.node = node
        self.qubit = qubit

    def invoke(self) -> None:
        self.node.handle(self)


class QubitReleasedEvent(Event):
    """
    Event sent by Forwarder to inform LinkLayer about a released (no longer needed) qubit.
    """

    def __init__(
        self,
        node: QNode,
        qubit: MemoryQubit,
        *,
        t: Time,
        name: str | None = None,
        by: Any = None,
    ):
        super().__init__(t=t, name=name, by=by)
        self.node = node
        self.qubit = qubit

    def invoke(self) -> None:
        self.node.handle(self)


class QubitEntangledEvent(Event):
    """
    Event sent by LinkLayer to notify Forwarder about new entangled qubit.
    """

    def __init__(
        self,
        node: QNode,
        neighbor: QNode,
        qubit: MemoryQubit,
        *,
        t: Time,
        name: str | None = None,
        by: Any = None,
    ):
        super().__init__(t=t, name=name, by=by)
        self.node = node
        self.neighbor = neighbor
        self.qubit = qubit

    def invoke(self) -> None:
        self.node.handle(self)
