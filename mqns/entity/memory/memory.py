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
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal, TypedDict, Unpack, overload, override

from mqns.entity.entity import Entity
from mqns.entity.memory.event import (
    MemoryReadRequestEvent,
    MemoryReadResponseEvent,
    MemoryWriteRequestEvent,
    MemoryWriteResponseEvent,
)
from mqns.entity.memory.memory_qubit import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.core import QuantumModel, QuantumModelT
from mqns.models.delay import DelayInput, parseDelay
from mqns.models.epr import Entanglement
from mqns.simulator import Event, Simulator


class QuantumMemoryInitKwargs(TypedDict, total=False):
    capacity: int
    delay: DelayInput
    t_cohere: float


class QuantumMemory(Entity):
    """Quantum memory stores qubits or entangled pairs

    It has two modes:
        Synchronous mode, users can use the ``read`` and ``write`` function to operate the memory directly without delay
        Asynchronous mode, users can use events to operate memories asynchronously
    """

    def __init__(self, name: str, **kwargs: Unpack[QuantumMemoryInitKwargs]):
        """
        Args:
            name: memory name.
            capacity: the capacity of this quantum memory, must be positive.
            delay: async read/write delay in seconds, or a ``DelayModel``.
            t_cohere: memory dephasing time in seconds, defaults to 1.0.
        """
        super().__init__(name=name)
        self.node: QNode
        """
        QNode that owns this memory.

        This is assigned by `QNode.set_memory()`.
        """
        self.capacity = kwargs.get("capacity", 1)
        """
        Memory capacity, i.e. how many qubits can be stored.
        Each qubit would have an address in `[0, capacity)`.
        """
        self.delay = parseDelay(kwargs.get("delay", 0))
        """Read/write delay, only applicable to async access."""

        self.t_cohere = kwargs.get("t_cohere", 1.0)

        assert self.capacity >= 1
        self._storage: list[tuple[MemoryQubit, QuantumModel | None]] = [
            (MemoryQubit(addr), None) for addr in range(self.capacity)
        ]
        self._usage = 0

        self._by_qchannel = dict[QuantumChannel, list[int]]()
        """
        Mapping from qchannel to assigned qubit addrs.
        Key is quantum channel assigned to qubits.
        Value is a sorted list of qubit addrs.
        """

    @override
    def install(self, simulator: Simulator) -> None:
        super().install(simulator)
        self.decoherence_delay = simulator.time(sec=self.t_cohere)
        """Memory dephasing time."""
        self.decoherence_rate = 1.0 / self.t_cohere
        """
        Memory dephasing rate in Hz.
        This is the inverse of memory dephasing time.
        EPR pair dephasing rate is the sum of memory dephasing rates.
        """

    @override
    def handle(self, event: Event) -> None:
        if isinstance(event, MemoryReadRequestEvent):
            result = self.read(event.key)  # will not update fidelity
            t = self.simulator.tc + self.delay.calculate()
            self.simulator.add_event(MemoryReadResponseEvent(self.node, result, request=event, t=t, by=self))
        elif isinstance(event, MemoryWriteRequestEvent):
            result = self.write(None, event.qubit)
            t = self.simulator.tc + self.delay.calculate()
            self.simulator.add_event(MemoryWriteResponseEvent(self.node, result, request=event, t=t, by=self))

    @property
    def count(self) -> int:
        """Return the quantity of stored qubits."""
        return self._usage

    @overload
    def find(
        self,
        predicate: Callable[[MemoryQubit, QuantumModel | None], bool],
        *,
        qchannel: QuantumChannel | None = None,
    ) -> Iterator[tuple[MemoryQubit, QuantumModel | None]]:
        pass

    @overload
    def find(
        self,
        predicate: Callable[[MemoryQubit, QuantumModelT], bool],
        *,
        qchannel: QuantumChannel | None = None,
        has: type[QuantumModelT],
    ) -> Iterator[tuple[MemoryQubit, QuantumModelT]]:
        pass

    def find(
        self,
        predicate: Callable[[MemoryQubit, Any], bool],
        *,
        qchannel: QuantumChannel | None = None,
        has: type[QuantumModelT] | None = None,
    ) -> Iterator[Any]:
        """
        Iterate over qubits and associated data that satisfy a predicate.

        Args:
            predicate: Callback function to accept or reject each qubit and associated data.
            qchannel: If set, only qubits assigned to specified quantum channel are considered.
            has: If set, only qubits with associated data of this type are considered.
        """
        iterable: Iterable[tuple[MemoryQubit, QuantumModel | None]] = self._storage
        if qchannel is not None:
            ch_addrs = self._by_qchannel.get(qchannel, [])
            iterable = (self._storage[addr] for addr in ch_addrs)
        for qubit, data in iterable:
            if (has is None or type(data) is has) and predicate(qubit, data):
                yield (qubit, data)

    def assign(self, ch: QuantumChannel, *, n=1) -> list[int]:
        """
        Assign n qubits to a particular quantum channel.

        This is only used at topology creation time.

        Returns:
            List of qubit addresses.

        Raises:
            OverflowError - insufficient unassigned qubits.
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
        self, ch: QuantumChannel, path_id: int, path_direction: PathDirection, *, n: int | Literal["all"] = 1
    ) -> list[int]:
        """
        Allocate n qubits to a given path ID.

        Args:
            ch: The quantum channel to which the memory qubit has been assigned.
            path_id: The identifier of the entanglement path to which the memory qubit will be allocated.
            path_direction: The end of the path to which the qubit allocated qubit points.
            n: Desired quantity, or "all" for all remaining qubits assigned to the channel.

        Returns:
            List of qubit addresses.

        Raises:
            OverflowError - insufficient unallocated qubits.
        """
        iterable = self.find(lambda q, _: q.path_id is None, qchannel=ch)
        if n == "all":
            want_all = True
        else:
            want_all = False
            iterable = itertools.islice(iterable, n)

        addrs: list[int] = []
        for qubit, _ in iterable:
            qubit.path_id = path_id
            qubit.path_direction = path_direction
            addrs.append(qubit.addr)

        if not want_all and len(addrs) != n:
            raise OverflowError(f"{self}: insufficient qubits for allocate({ch},n={n})")
        return addrs

    def deallocate(self, *addrs: int) -> None:
        """
        Deallocate one or more qubits from any assigned path.

        This method finds the memory qubit with the given address and clears its
        path assignment (i.e., resets its `path_id` to None). It does not modify the
        quantum state or remove the qubit from memory.
        """
        for addr in addrs:
            qubit, _ = self._storage[addr]
            qubit.path_id = None
            qubit.path_direction = None

    @overload
    def read(self, key: int | str, *, remove: bool | QuantumModel = False) -> tuple[MemoryQubit, QuantumModel | None] | None:
        """
        Retrieve a qubit and associated data.

        Args:
            key: Qubit address or EPR name.
            remove: Whether to remove the data.
                    If specified as QuantumModel, remove only if stored data is the same object.

        Returns:
            Qubit and associated data (possibly empty), or None if qubit is not found by EPR name.

        Raises:
            IndexError - qubit address out of range.
        """
        pass

    @overload
    def read(
        self, key: int | str, *, must: Literal[True], remove: bool | QuantumModel = False
    ) -> tuple[MemoryQubit, QuantumModel | None]:
        """
        Retrieve a qubit and associated data.

        Args:
            key: Qubit address or EPR name.
            must: True.
            remove: Whether to remove the data.
                    If specified as QuantumModel, remove only if stored data is the same object.

        Returns:
            Qubit and associated data (possibly empty).

        Raises:
            IndexError - qubit not found.
            ValueError - no quantum information is stored.
        """
        pass

    @overload
    def read(
        self,
        key: int | str,
        *,
        must: Literal[True] = True,
        has: type[QuantumModelT],
        set_fidelity=False,
        remove: bool | QuantumModel = False,
    ) -> tuple[MemoryQubit, QuantumModelT]:
        """
        Retrieve a qubit and associated data.

        Args:
            key: Qubit address or EPR name.
            must: True (implied).
            has: Expected type of stored data.
            set_fidelity: Whether to update fidelity, XXX current formula is inaccurate for EPRs.
            remove: Whether to remove the data.
                    If specified as QuantumModel, remove only if stored data is the same object.

        Returns:
            Qubit and associated data (has type specified in `must`).

        Raises:
            IndexError - qubit not found.
            ValueError - no quantum information is stored or it is not the expected type.
        """
        pass

    def read(
        self,
        key: int | str,
        *,
        must=False,
        has: type[QuantumModelT] | None = None,
        set_fidelity=False,
        remove: bool | QuantumModel = False,
    ):
        if type(key) is int:
            qubit, data = self._storage[key]
        else:
            qubit, data = next(self.find(lambda _, v: getattr(v, "name", None) == key), (None, None))

        if qubit is None:
            if must or has:
                raise IndexError(f"{self}: cannot find {key}")
            return None

        if has and type(data) is not has:
            raise ValueError(f"{self}: data at {qubit.addr} is not {has}")

        if set_fidelity and isinstance(data, Entanglement) and not data.read:
            data.read = True
            now = self.simulator.tc
            data.store_error_model((now - data.creation_time).sec, self.decoherence_rate)

        if remove in (True, data):
            qubit.set_event(QuantumMemory, None)  # cancel scheduled decoherence event
            self._usage -= 1
            self._storage[qubit.addr] = (qubit, None)

        return qubit, data

    def write(self, key: int | str | None, data: QuantumModel, *, replace=False) -> MemoryQubit:
        """
        Store data in memory.

        Args:
            key: Qubit address, `qubit.active` identifier, or `None` for any unused qubit.
            data: Data to be stored.
                  If this is an EPR, a decoherence event is scheduled automatically.
            replace: True allows replacing existing data; False requires qubit to be empty.

        Returns:
            Qubit where the data is stored.

        Raises:
            IndexError - qubit not found by `key` or no qubit available.
            ValueError - `replace=False` but qubit has existing data.
        """
        if type(key) is int:
            qubit, old = self._storage[key]
        elif type(key) is str:
            qubit, old = next(self.find(lambda q, _: q.active == key), (None, None))
        else:
            qubit, old = next(self.find(lambda _, v: v is None), (None, None))

        if qubit is None:
            raise IndexError("qubit not found")

        if not replace and old is not None:
            raise ValueError(f"qubit contains existing data: {old}")

        self._storage[qubit.addr] = (qubit, data)
        if old is None:
            self._usage += 1

        if isinstance(data, Entanglement):
            self._schedule_decohere(qubit, data)
        elif old is not None:
            qubit.set_event(QuantumMemory, None)  # cancel old decoherence event

        return qubit

    def clear(self) -> None:
        """Clear all qubits in the memory."""
        for qubit, _ in self._storage:
            qubit.reset_state()
            self._storage[qubit.addr] = (qubit, None)
        self._usage = 0

    def _schedule_decohere(self, qubit: MemoryQubit, epr: Entanglement):
        from mqns.network.protocol.event import QubitDecoheredEvent  # noqa: PLC0415

        assert epr.decoherence_time >= self.simulator.tc

        event = QubitDecoheredEvent(self, qubit, epr, t=epr.decoherence_time)
        qubit.set_event(QuantumMemory, event)
        self.simulator.add_event(event)

    def handle_decohere_qubit(self, qubit: MemoryQubit, epr: Entanglement) -> bool:
        """
        Part of `QubitDecoheredEvent` logic.

        Args:
            qubit: The memory qubit that has reached its decoherence time.
            epr: The associated entanglement.

        Returns:
            Whether the event should be dispatched to inform LinkLayer.
        """

        epr.is_decoherenced = True

        _, new_qm = self.read(qubit.addr, must=True, remove=epr)
        if new_qm is not epr:
            # qubit already released via swap/purify or re-entangled
            return False

        qubit.state = QubitState.RELEASE
        return True

    def __repr__(self) -> str:
        return "<memory " + self.name + ">"
