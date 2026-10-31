#    Modified by Amar Abane for Multiverse Quantum Network Simulator
#    Date: 05/17/2025
#    Summary of changes: Adapted logic to support dynamic approaches.
#
#    This file is based on a snapshot of SimQN (https://github.com/QNLab-USTC/SimQN),
#    which is licensed under the GNU General Public License v3.0.
#
#    The original SimQN header is included below.


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

import heapq
import itertools
from collections.abc import Callable, Container, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, Unpack, overload, override

from mqns.entity.entity import Entity
from mqns.entity.memory.event import MemoryDecohereEvent
from mqns.entity.memory.memory_qubit import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.core import QuantumModel
from mqns.models.epr import Entanglement
from mqns.models.error import TimeDecayInput, parse_time_decay
from mqns.simulator import EventDispatcherMixin, Simulator, event_handler
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.entity.node import QNode


class QuantumMemoryInitKwargs(TypedDict, total=False):
    """QuantumMemory constructor parameters."""

    capacity: int
    """How many qubits can be stored in memory, must be positive, defaults to 1."""
    t_cohere: float
    """Memory decoherence time in seconds, defaults to 1."""
    time_decay: TimeDecayInput
    """Time decay function for loss of quantum information, defaults to dephasing in ``t_cohere``."""


class QuantumMemory(EventDispatcherMixin, Entity):
    """
    Quantum memory stores qubits or entangled pairs.
    """

    node: QNode
    """
    QNode that owns this memory.

    This is assigned by ``QNode.memory`` setter.
    """

    @staticmethod
    def check_leaks(
        nodes: Iterable["QNode"],
        *,
        states: Container[QubitState] = (QubitState.RAW, QubitState.RELEASE),
        deallocated=False,
        unassigned=False,
    ) -> None:
        """
        Verify that all MemoryQubits on every node are released.

        Args:
            nodes: List of quantum nodes, such as ``QuantumNetwork.nodes``.
            states: Acceptable states.
            deallocated: If True, qubit must not be allocated to a path.
            unassigned: If True, qubit must not be assigned to a channel.

        Raises:
            MemoryError: Some qubits have unacceptable state.
        """
        __tracebackhide__ = True
        errors: list[str] = []
        for node in nodes:
            for mq, data in node.memory.find(QuantumMemory.predicate_all):
                if mq.state not in states:
                    errors.append(f"{node.name} {mq} has unexpected state {mq.state} | {data}")
                if deallocated and mq.path_id is not None:
                    errors.append(f"{node.name} {mq} is allocated to path {mq.path_id}")
                if unassigned and mq.qchannel:
                    errors.append(f"{node.name} {mq} is assigned to {mq.qchannel}")
        if n := len(errors):
            log.warning("QuantumMemory.check_leaks() found %s errors:\n\t%s", n, "\n\t".join(errors))
            raise MemoryError(*errors)

    def __init__(self, name: str, **kwargs: Unpack[QuantumMemoryInitKwargs]):
        """
        Constructor.

        Args:
            name: Memory entity name.
        """
        super().__init__(name=name)

        self.capacity: Final = kwargs.get("capacity", 1)
        """
        Memory capacity, i.e. how many qubits can be stored.
        Each qubit has an address in ``[0, capacity)``.
        """

        self._t_cohere_input = kwargs.get("t_cohere", 1.0)
        self._time_decay_input = kwargs.get("time_decay")

        assert self.capacity >= 1
        self._storage: list[tuple[MemoryQubit, QuantumModel | None]] = [
            (MemoryQubit(addr), None) for addr in range(self.capacity)
        ]
        self._usage = 0

        self._by_qchannel: dict[QuantumChannel, list[int]] = {}
        """
        Mapping from qchannel to assigned qubit addrs.
        Key is quantum channel assigned to qubits.
        Value is a sorted list of qubit addrs.
        """

    @override
    def install(self, simulator: Simulator) -> None:
        super().install(simulator)

        self.t_cohere = simulator.time(sec=self._t_cohere_input)
        """
        Memory decoherence time, often known as T2.

        Stored qubits trigger ``MemoryDecohereEvent`` at this timer.
        """

        self.time_decay = parse_time_decay(self._time_decay_input, self.t_cohere)
        """Time based decay function constructed from store error model."""

    @event_handler
    def handle_decohere(self, event: MemoryDecohereEvent) -> None:
        mq = event.qubit
        qm = event.qm
        if isinstance(qm, Entanglement):
            qm.is_decohered = True

        # Both .read(remove=qm) and .state=RELEASE would cancel attached events.
        # Pop the current MemoryDecohereEvent so that it can be dispatched to applications.
        mq.events.pop(MemoryDecohereEvent)

        _, new_qm = self.read(mq.addr, remove=qm)
        if new_qm is not qm:
            # qubit already released via swap/purify or re-entangled
            return

        mq.state = QubitState.RELEASE
        self.node.handle(event)

    @property
    def count(self) -> int:
        """Return the quantity of stored qubits."""
        return self._usage

    @staticmethod
    def predicate_all(mq: MemoryQubit, qm: QuantumModel | None) -> bool:
        """
        A predicate for ``find()`` that accepts all qubits.
        """
        _ = mq, qm
        return True

    @overload
    def find(
        self,
        predicate: Callable[[MemoryQubit, QuantumModel | None], bool],
        *,
        qchannel: QuantumChannel | None = None,
    ) -> Iterator[tuple[MemoryQubit, QuantumModel | None]]: ...

    @overload
    def find[M: QuantumModel](
        self,
        predicate: Callable[[MemoryQubit, M], bool],
        *,
        qchannel: QuantumChannel | None = None,
        has: type[M],
    ) -> Iterator[tuple[MemoryQubit, M]]: ...

    def find[M: QuantumModel](
        self,
        predicate: Callable[[MemoryQubit, Any], bool],
        *,
        qchannel: QuantumChannel | None = None,
        has: type[M] | None = None,
    ) -> Iterator[Any]:
        """
        Iterate over qubits and associated data that satisfy a predicate.

        Args:
            predicate: Callback function to accept or reject each qubit and associated data.
                Use ``QuantumMemory.predicate_all`` to accept all qubits.
            qchannel: If set, only qubits assigned to specified quantum channel are considered.
            has: If set, only qubits with associated data of this type are considered.
        """
        iterable: Iterable[tuple[MemoryQubit, QuantumModel | None]] = self._storage
        if qchannel is not None:
            ch_addrs = self._by_qchannel.get(qchannel, [])
            iterable = (self._storage[addr] for addr in ch_addrs)
        for qubit, data in iterable:
            if (has is None or type(data) is has) and predicate(qubit, data):
                yield qubit, data

    def assign(self, ch: QuantumChannel, *, n=1) -> list[int]:
        """
        Assign n qubits to a particular quantum channel.

        This is only used at topology creation time.

        Returns:
            List of qubit addresses.

        Raises:
            OverflowError: insufficient unassigned qubits.
        """
        addrs: list[int] = []
        for qubit, _ in itertools.islice(self.find(lambda q, _: q.qchannel is None), n):
            qubit.qchannel = ch
            addrs.append(qubit.addr)

        if len(addrs) != n:
            raise OverflowError(f"{self}: insufficient qubits for assign(n={n})")

        self._by_qchannel[ch] = list(heapq.merge(self._by_qchannel.get(ch, []), addrs))
        return addrs

    def unassign(self, *addrs: int) -> None:
        """
        Unassign one or more qubits from any quantum channel.
        """
        for addr in addrs:
            qubit, _ = self._storage[addr]
            if qubit.qchannel is None:
                continue

            ch_addrs = self._by_qchannel[qubit.qchannel]
            ch_addrs.remove(addr)
            if len(ch_addrs) == 0:
                del self._by_qchannel[qubit.qchannel]

            qubit.qchannel = None

    def allocate(
        self,
        ch: QuantumChannel,
        path_id: int,
        path_direction: PathDirection,
        *,
        n: int = 1,
    ) -> list[int]:
        """
        Allocate n qubits to a given path ID.

        Args:
            ch: Only consider qubits assigned to this quantum channel.
            path_id: Allocate qubits to this path identifier, retrievable via ``MemoryQubit.path_id``.
            path_direction: Allocate qubits to this path direction, retrievable via ``MemoryQubit.path_direction``.
            n: Desired quantity.

        Returns:
            List of qubit addresses.

        Raises:
            OverflowError: insufficient unallocated qubits.
        """
        qubits = list(itertools.islice(self.find(lambda q, _: q.path_id is None, qchannel=ch), n))
        if len(qubits) != n:
            raise OverflowError(f"{self}: insufficient qubits for allocate({ch},n={n})")

        addrs: list[int] = []
        for mq, _ in qubits:
            mq.path_id = path_id
            mq.path_direction = path_direction
            addrs.append(mq.addr)
        return addrs

    def deallocate(self, *addrs: int) -> None:
        """
        Deallocate one or more qubits from any assigned path.

        If a qubit is currently occupied, the deallocation takes effect when it next reaches RAW state.
        """
        for addr in addrs:
            mq, _ = self._storage[addr]
            mq.on_raw(QuantumMemory._deallocate_qubit)

    @staticmethod
    def _deallocate_qubit(mq: MemoryQubit) -> None:
        mq.path_id = None
        mq.path_direction = None

    @overload
    def read(
        self, key: int, *, must: Literal[True] = True, remove: bool | QuantumModel = False
    ) -> tuple[MemoryQubit, QuantumModel | None]:
        """
        Retrieve a qubit and associated data.

        Args:
            key: Qubit address.
            remove: Whether to remove the data.
                    If specified as QuantumModel, remove only if stored data is the same object.

        Returns:
            Qubit and associated data (possibly empty).
        """

    @overload
    def read(self, key: str, *, remove: bool | QuantumModel = False) -> tuple[MemoryQubit, QuantumModel | None] | None:
        """
        Retrieve a qubit and associated data.

        Args:
            key: Qubit reservation key.
            remove: Whether to remove the data.
                    If specified as QuantumModel, remove only if stored data is the same object.

        Returns:
            Qubit and associated data (possibly empty), or None if qubit is not found by reservation key.
        """

    @overload
    def read(
        self, key: str, *, must: Literal[True], remove: bool | QuantumModel = False
    ) -> tuple[MemoryQubit, QuantumModel | None]:
        """
        Retrieve a qubit and associated data.

        Args:
            key: Qubit address or reservation key.
            must: True.
            remove: Whether to remove the data.
                    If specified as QuantumModel, remove only if stored data is the same object.

        Returns:
            Qubit and associated data (possibly empty).

        Raises:
            LookupError: Qubit not found.
        """

    @overload
    def read[M: QuantumModel](
        self,
        key: int | str,
        *,
        must: Literal[True] = True,
        has: type[M],
        remove: bool | QuantumModel = False,
    ) -> tuple[MemoryQubit, M]:
        """
        Retrieve a qubit and associated data.

        Args:
            key: Qubit address or reservation key.
            must: True (implied).
            has: Expected type of stored data.
            remove: Whether to remove the data.
                    If specified as QuantumModel, remove only if stored data is the same object.

        Returns:
            Qubit and associated data (has type specified in ``has``).

        Raises:
            LookupError: Qubit not found.
            ValueError: No quantum information is stored or it is not the expected type.
        """

    def read[M: QuantumModel](
        self,
        key: int | str,
        *,
        must=False,
        has: type[M] | None = None,
        remove: bool | QuantumModel = False,
    ):
        if type(key) is int:
            qubit, data = self._storage[key]
        else:
            qubit, data = next(self.find(lambda q, _: q.key == key), (None, None))
            if qubit is None:
                if must or has:
                    raise LookupError(f"{self}: cannot find {key}")
                return None

        if has and type(data) is not has:
            raise ValueError(f"{self}: data at {qubit.addr} is not {has}")

        if data is not None and remove in (True, data):
            qubit.events.discard(MemoryDecohereEvent)
            self._usage -= 1
            self._storage[qubit.addr] = qubit, None

        return qubit, data

    def write(self, key: int | str, data: QuantumModel, *, replace=False) -> MemoryQubit:
        """
        Store data in memory.

        Args:
            key: Qubit address or reservation key.
            data: Data to be stored.
                  If this is an EPR, a decoherence event is scheduled automatically.
            replace: True allows replacing existing data; False requires qubit to be empty.

        Returns:
            Qubit where the data is stored.

        Raises:
            LookupError: qubit not found by ``key`` or no qubit available.
            ValueError: ``replace=False`` but qubit has existing data.
        """
        if type(key) is int:
            qubit, old = self._storage[key]
        else:
            qubit, old = next(self.find(lambda q, _: q.key == key), (None, None))
            if qubit is None:
                raise LookupError(f"{self}: qubit {key} not found")

        if old is None:
            self._usage += 1
        elif not replace:
            raise ValueError(f"{self}: {qubit} contains existing data: {old}")

        self._storage[qubit.addr] = qubit, data

        if isinstance(data, Entanglement):
            assert data.decohere_time >= self.simulator.tc
            self.simulator.sched(event := MemoryDecohereEvent(self, qubit, data, t=data.decohere_time))
            qubit.events.add(event)
        elif old is not None:
            qubit.events.discard(MemoryDecohereEvent)

        return qubit

    def clear(self) -> None:
        """Clear all qubits in the memory."""
        for qubit, _ in self._storage:
            qubit.reset_state(QubitState.RAW)
            self._storage[qubit.addr] = qubit, None
        self._usage = 0
