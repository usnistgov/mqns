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

from collections.abc import Iterable
from typing import ClassVar, final, override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.simulator import Event, Time


@final
class LLActivateEvent(Event):
    """
    Instruct LinkLayer to start entanglement generation.
    """

    def __init__(
        self,
        node: QNode,
        ch: QuantumChannel,
        *,
        t: Time,
        gid: int,
        is_primary: bool,
        qubits: Iterable[MemoryQubit],
    ):
        """
        Args:
            node: Node with :py:class:`~mqns.network.ll.LinkLayer` application.
            ch: Quantum channel for entanglement generation.
            gid: Group identifier, must be unique per channel and same on both sides.
            is_primary: True for primary role, False for secondary role.
                Two nodes on the same channel must take opposite roles.
                The choice must be consistent for all groups on the same channel.
            qubits: Qubits allocated for the group.
        """
        self.node = node
        self.peer = ch.find_peer(node)
        super().__init__(t, f"node={node.name}, role={'pri' if is_primary else '2nd'}, peer={self.peer.name}, gid={gid}")
        self.ch = ch
        self.gid = gid
        self.is_primary = is_primary
        self.qubits = list(qubits)

    @override
    def invoke(self) -> None:
        self.node.handle(self)


@final
class LLDeactivateEvent(Event):
    """
    Instruct LinkLayer to stop entanglement generation.
    """

    def __init__(
        self,
        node: QNode,
        ch: QuantumChannel,
        *,
        t: Time,
        gid: int,
    ):
        """
        Args:
            node: Node with :py:class:`~mqns.network.ll.LinkLayer` application.
            ch: Quantum channel for entanglement generation.
            gid: Group identifier, must be unique per channel and same on both sides.
        """
        self.node = node
        self.peer = ch.find_peer(node)
        super().__init__(t, f"node={node.name}, peer={self.peer.name}, gid={gid}")
        self.ch = ch
        self.gid = gid

    @override
    def invoke(self) -> None:
        self.node.handle(self)


@final
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
    ):
        super().__init__(t, f"{node.name}-{neighbor.name}, key={qubit.key}, addr={qubit.addr}")
        self.node = node
        self.neighbor = neighbor
        self.qubit = qubit
        assert self.qubit.state is QubitState.ENTANGLED0, f"unexpected state {qubit.state}"

    @override
    def invoke(self) -> None:
        self.qubit.state = QubitState.ENTANGLED1
        self.node.handle(self)


@final
class QubitConsumeEvent(Event):
    """
    Event sent by Forwarder to notify Consumer about end-to-end entanglement.
    """

    def __init__(
        self,
        node: QNode,
        qubit: MemoryQubit,
        epr: Entanglement,
        *,
        t: Time,
        req_id: int,
    ):
        super().__init__(t, f"{qubit.addr}, {epr.name}")
        self.node = node
        self.qubit = qubit
        self.epr = epr
        self.req_id = req_id
        assert qubit.state is QubitState.CONSUME, f"unexpected state {qubit.state}"

    @override
    def invoke(self) -> None:
        assert self.qubit.state is QubitState.CONSUME, f"unexpected state {self.qubit.state}"
        self.node.handle(self)


@final
class QubitReleasedEvent(Event):
    """
    Event sent by Forwarder/Consumer to inform LinkLayer about a released (no longer needed) qubit.
    """

    REASON_STR: ClassVar = {True: "decoh", False: "release"}

    def __init__(
        self,
        node: QNode,
        qubit: MemoryQubit,
        *,
        t: Time,
        is_decoh=False,
    ):
        super().__init__(t, f"addr={qubit.addr} key={qubit.key} {self.REASON_STR[is_decoh]}")
        self.node = node
        self.qubit = qubit
        self.is_decoh = is_decoh
        assert self.qubit.state is QubitState.RELEASE, f"unexpected state {self.qubit.state}"

    @override
    def invoke(self) -> None:
        assert self.qubit.state is QubitState.RELEASE, f"unexpected state {self.qubit.state}"
        self.node.handle(self)
