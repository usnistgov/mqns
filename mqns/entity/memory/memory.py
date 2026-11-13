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

from collections.abc import Callable, Container, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, Unpack, overload, override

from mqns.entity.entity import Entity
from mqns.entity.memory.event import MemoryDecohereEvent
from mqns.entity.memory.memory_qubit import MemoryQubit, QubitState
from mqns.entity.node import QNode
from mqns.models.core import QuantumModel
from mqns.models.epr import Entanglement
from mqns.models.error import TimeDecayFunc, TimeDecayInput, parse_time_decay
from mqns.simulator import EventDispatcherMixin, Simulator, Time, event_handler
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

    This is assigned by :py:attr:`QNode.memory <mqns.entity.node.QNode.memory>` setter.
    """

    @staticmethod
    def check_leaks(
        nodes: Iterable["QNode"],
        *,
        states: Container[QubitState] = (QubitState.RAW, QubitState.RELEASE),
    ) -> None:
        """
        Verify that all MemoryQubits on every node are released.

        Args:
            nodes: List of quantum nodes, such as :py:attr:`QuantumNetwork.nodes <mqns.network.network.QuantumNetwork.nodes>`.
            states: Acceptable states.

        Raises:
            MemoryError: Some qubits have unacceptable state.
        """
        __tracebackhide__ = True
        errors: list[str] = []
        for node in nodes:
            for mq, data in node.memory._storage:
                if mq.state not in states:
                    errors.append(f"{node.name} {mq} has unexpected state {mq.state} | {data}")
        if n := len(errors):
            log.warning("QuantumMemory.check_leaks() found %s errors:\n\t%s", n, "\n\t".join(errors))
            raise MemoryError(*errors)

    capacity: Final[int]
    """
    Memory capacity, i.e. how many qubits can be stored.
    Each qubit has an address in ``[0, capacity)``.
    """

    t_cohere: Time
    """
    Memory decoherence time, often known as T2.

    Stored qubits trigger :py:class:`MemoryDecohereEvent` at this timer.
    """

    time_decay: TimeDecayFunc
    """Time based decay function constructed from store error model."""

    def __init__(self, name: str, **kwargs: Unpack[QuantumMemoryInitKwargs]):
        """
        Constructor.

        Args:
            name: Memory entity name.
        """
        super().__init__(name)

        self.capacity = kwargs.get("capacity", 1)

        self._t_cohere_input = kwargs.get("t_cohere", 1.0)
        self._time_decay_input = kwargs.get("time_decay")

        assert self.capacity >= 1
        self._storage: list[tuple[MemoryQubit, QuantumModel | None]] = [
            (MemoryQubit(addr), None) for addr in range(self.capacity)
        ]
        self._usage = 0

    @override
    def install(self, simulator: Simulator) -> None:
        super().install(simulator)

        self.t_cohere = simulator.time(sec=self._t_cohere_input)
        self.time_decay = parse_time_decay(self._time_decay_input, self.t_cohere)

    @event_handler
    def handle_decohere(self, event: MemoryDecohereEvent) -> None:
        """
        Handle memory decoherence event locally, and then re-dispatch to the applications.
        """
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

    @overload
    def find(
        self,
        predicate: Callable[[MemoryQubit, QuantumModel | None], bool],
    ) -> Iterator[tuple[MemoryQubit, QuantumModel | None]]: ...

    @overload
    def find[M: QuantumModel](
        self,
        predicate: Callable[[MemoryQubit, M], bool],
        *,
        has: type[M],
    ) -> Iterator[tuple[MemoryQubit, M]]: ...

    def find[M: QuantumModel](
        self,
        predicate: Callable[[MemoryQubit, Any], bool],
        *,
        has: type[M] | None = None,
    ) -> Iterator[Any]:
        """
        Iterate over qubits and associated data that satisfy a predicate.

        Args:
            predicate: Callback function to accept or reject each qubit and associated data.
            has: If set, only qubits with associated data of this type are considered.
        """
        for qubit, data in self._storage:
            if (has is None or type(data) is has) and predicate(qubit, data):
                yield qubit, data

    @overload
    def read(
        self, key: int, *, must: Literal[True] = True, remove: bool | QuantumModel = False
    ) -> tuple[MemoryQubit, QuantumModel | None]: ...

    @overload
    def read(self, key: str, *, remove: bool | QuantumModel = False) -> tuple[MemoryQubit, QuantumModel | None] | None: ...

    @overload
    def read(
        self, key: str, *, must: Literal[True], remove: bool | QuantumModel = False
    ) -> tuple[MemoryQubit, QuantumModel | None]: ...

    @overload
    def read[M: QuantumModel](
        self,
        key: int | str,
        *,
        must: Literal[True] = True,
        has: type[M],
        remove: bool | QuantumModel = False,
    ) -> tuple[MemoryQubit, M]: ...

    def read[M: QuantumModel](
        self,
        key: int | str,
        *,
        must=False,
        has: type[M] | None = None,
        remove: bool | QuantumModel = False,
    ):
        """
        Retrieve a qubit and associated data.

        Args:
            key: Qubit address or reservation key.
            must: When the qubit is not found by reservation key, raise LookupError if ``must=True``,
                return None otherwise. Out-of-range qubit address always raises LookupError.
            has: Expected type of stored data; implies ``must=True``.
            remove: Whether to remove the data.
                    If specified as QuantumModel, remove only if stored data is the same object.

        Returns:
            Qubit and associated data.

        Raises:
            LookupError: Qubit not found.
            ValueError: When ``must=True``, quantum information is absent;
                when ``has`` is specified, stored quantum information does not match the type.
        """

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
                  If this is an :py:class:`~mqns.models.epr.Entanglement`, :py:class:`MemoryDecohereEvent`
                  is scheduled after :py:attr:`t_cohere`.
            replace: True allows replacing existing data; False requires qubit to be empty.

        Returns:
            Qubit where the data is stored.

        Raises:
            LookupError: Qubit not found.
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
