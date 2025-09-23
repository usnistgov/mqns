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

from typing import Any

from typing_extensions import override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import WernerStateEntanglement
from mqns.simulator import Event, Time


class ManageActiveChannels(Event):
    """
    Event sent by Forwarder to request LinkLayer to start/stop generating EPRs over a qchannel.
    """

    def __init__(
        self,
        node: QNode,
        neighbor: QNode,
        qchannel: QuantumChannel,
        *,
        path_id: int | None = None,
        start: bool,
        t: Time,
        name: str | None = None,
        by: Any = None,
    ):
        super().__init__(t, name, by)
        self.node = node
        self.neighbor = neighbor
        self.qchannel = qchannel
        self.path_id = path_id
        self.start = start

    @override
    def invoke(self) -> None:
        self.node.handle(self)


class LinkArchSuccessEvent(Event):
    """
    Event in LinkLayer to notify itself or its neighbor about successful entanglement in link architecture.
    """

    def __init__(
        self,
        node: QNode,
        epr: WernerStateEntanglement,
        *,
        t: Time,
        name: str | None = None,
        by: Any = None,
    ):
        super().__init__(t, name, by)
        self.node = node
        self.epr = epr

    @override
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
        super().__init__(t, name, by)
        self.node = node
        self.neighbor = neighbor
        self.qubit = qubit
        assert self.qubit.state == QubitState.ENTANGLED0

    @override
    def invoke(self) -> None:
        self.qubit.state = QubitState.ENTANGLED1
        self.node.handle(self)


class QubitDecoheredEvent(Event):
    """
    Event sent by Memory to inform LinkLayer about a decohered qubit.
    """

    def __init__(self, node: QNode, qubit: MemoryQubit, *, t: Time, name: str | None = None, by: Any = None):
        super().__init__(t, name, by)
        self.node = node
        self.qubit = qubit
        assert self.qubit.state == QubitState.RELEASE

    @override
    def invoke(self) -> None:
        assert self.qubit.state == QubitState.RELEASE
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
        super().__init__(t, name, by)
        self.node = node
        self.qubit = qubit
        assert self.qubit.state == QubitState.RELEASE

    @override
    def invoke(self) -> None:
        assert self.qubit.state == QubitState.RELEASE
        self.node.handle(self)
