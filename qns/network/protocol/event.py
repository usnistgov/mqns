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
from typing import TYPE_CHECKING, Any

from qns.entity.memory.memory_qubit import MemoryQubit
from qns.entity.node.qnode import QNode
from qns.simulator.event import Event
from qns.simulator.ts import Time

if TYPE_CHECKING:
    from qns.network.protocol.link_layer import LinkLayer
    from qns.network.protocol.proactive_forwarder import ProactiveForwarder

class TypeEnum(Enum):
    ADD = auto()
    REMOVE = auto()

class ManageActiveChannels(Event):
    """``ManageActiveChannels`` is the event sent by the forwarder to request to start generating EPRs over a qchannel
    """

    def __init__(self, link_layer: "LinkLayer", neighbor: QNode, type: TypeEnum,
                 t: Time | None = None, name: str | None = None, by: Any | None = None):
        super().__init__(t=t, name=name, by=by)
        self.link_layer = link_layer
        self.neighbor = neighbor
        self.type = type

    def invoke(self) -> None:
        self.link_layer.handle_event(self)


class QubitDecoheredEvent(Event):
    """``QubitDecoheredEvent`` is the event that informs LinkLayer about a decohered qubit from Memory
    """

    def __init__(self, link_layer: "LinkLayer", qubit: MemoryQubit,
                 t: Time | None = None, name: str | None = None,
                 by: Any | None = None):
        super().__init__(t=t, name=name, by=by)
        self.link_layer = link_layer
        self.qubit = qubit

    def invoke(self) -> None:
        self.link_layer.handle_event(self)


class QubitReleasedEvent(Event):
    """``QubitReleasedEvent`` is the event that informs LinkLayer about a released qubit from NetworkLayer
    """

    def __init__(self, link_layer: "LinkLayer", qubit: MemoryQubit, e2e: bool = False,
                 t: Time | None = None, name: str | None = None,
                 by: Any | None = None):
        super().__init__(t=t, name=name, by=by)
        self.link_layer = link_layer
        self.qubit = qubit
        self.e2e = e2e

    def invoke(self) -> None:
        self.link_layer.handle_event(self)


class QubitEntangledEvent(Event):
    """``QubitEntangledEvent`` is the event that notifies NetworkLayer about new entangled qubit from LinkLayer
    """

    def __init__(self, forwarder: "ProactiveForwarder", neighbor: QNode, qubit: MemoryQubit,
                 t: Time | None = None, name: str | None = None,
                 by: Any | None = None):
        super().__init__(t=t, name=name, by=by)
        self.forwarder = forwarder
        self.neighbor = neighbor
        self.qubit = qubit

    def invoke(self) -> None:
        self.forwarder.handle_event(self)
