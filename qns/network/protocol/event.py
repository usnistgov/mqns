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

from enum import Enum, auto
from typing import Any, Optional, Union
from qns.entity.node.qnode import QNode
from qns.simulator.event import Event
from qns.simulator.ts import Time
from qns.network.protocol.link_layer import LinkLayer
from qns.network.protocol.proactive_routing import ProactiveRouting
from qns.entity.memory.memory_qubit import MemoryQubit

class TypeEnum(Enum):
    ADD = auto()
    REMOVE = auto()

class LinkLayerManageActiveChannels(Event):
    """
    ``LinkLayerManageActiveChannels`` is the event that request to start EPR generation over a qchannel
    """
    def __init__(self, link_layer: LinkLayer, next_hop: QNode, type: TypeEnum, 
                 t: Optional[Time] = None, name: Optional[str] = None, by: Optional[Any] = None):
        super().__init__(t=t, name=name, by=by)
        self.link_layer = link_layer
        self.next_hop = next_hop
        self.type = type

    def invoke(self) -> None:
        self.link_layer.handle_event(self)


class QubitDecoheredEvent(Event):
    """
    ``QubitDecoheredEvent`` is the event that informs LinkLayer about a decohered qubit from Memory
    """
    def __init__(self, link_layer: LinkLayer, qubit: MemoryQubit, 
                 t: Optional[Time] = None, name: Optional[str] = None,
                 by: Optional[Any] = None):
        super().__init__(t=t, name=name, by=by)
        self.link_layer = link_layer
        self.qubit = qubit

    def invoke(self) -> None:
        self.link_layer.handle_event(self)
        
        
class QubitReleasedEvent(Event):
    """
    ``QubitReleasedEvent`` is the event that informs LinkLayer about a released qubit from NetworkLayer
    """
    def __init__(self, link_layer: LinkLayer, qubit: MemoryQubit, 
                 t: Optional[Time] = None, name: Optional[str] = None,
                 by: Optional[Any] = None):
        super().__init__(t=t, name=name, by=by)
        self.link_layer = link_layer
        self.qubit = qubit

    def invoke(self) -> None:
        self.link_layer.handle_event(self)


class QubitEntangledEvent(Event):
    """
    ``QubitEntangledEvent`` is the event that notifies NetworkLayer about new entangled qubit from LinkLayer
    """
    def __init__(self, net_layer: ProactiveRouting, neighbor: QNode, qubit: MemoryQubit, 
                 t: Optional[Time] = None, name: Optional[str] = None,
                 by: Optional[Any] = None):
        super().__init__(t=t, name=name, by=by)
        self.net_layer = net_layer
        self.neighbor = neighbor
        self.qubit = qubit

    def invoke(self) -> None:
        self.net_layer.handle_event(self)
