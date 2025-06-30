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
from typing import cast

from qns.entity.memory import MemoryQubit, QuantumMemory
from qns.entity.node import Application, QNode
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
        *,
        neighbor: QNode,
        type: TypeEnum,
        t: Time,
        name: str | None = None,
        by: Application,
    ):
        super().__init__(t=t, name=name, by=by)
        self.neighbor = neighbor
        self.type = type

    def invoke(self) -> None:
        cast(Application, self.by).get_node().handle(self)


class QubitDecoheredEvent(Event):
    """
    Event sent by Memory to inform LinkLayer about a decohered qubit.
    """

    def __init__(self, *, qubit: MemoryQubit, t: Time, name: str | None = None, by: QuantumMemory):
        super().__init__(t=t, name=name, by=by)
        self.qubit = qubit

    def invoke(self) -> None:
        cast(QNode, self.by.node).handle(self)


class QubitReleasedEvent(Event):
    """
    Event sent by Forwarder to inform LinkLayer about a released (no longer needed) qubit.
    """

    def __init__(
        self,
        *,
        qubit: MemoryQubit,
        e2e: bool = False,
        t: Time,
        name: str | None = None,
        by: Application,
    ):
        super().__init__(t=t, name=name, by=by)
        self.qubit = qubit
        self.e2e = e2e

    def invoke(self) -> None:
        cast(Application, self.by).get_node().handle(self)


class QubitEntangledEvent(Event):
    """
    Event sent by LinkLayer to notify Forwarder about new entangled qubit.
    """

    def __init__(
        self,
        *,
        neighbor: QNode,
        qubit: MemoryQubit,
        t: Time,
        name: str | None = None,
        by: Application,
    ):
        super().__init__(t=t, name=name, by=by)
        self.neighbor = neighbor
        self.qubit = qubit

    def invoke(self) -> None:
        cast(Application, self.by).get_node().handle(self)
