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
from typing import TYPE_CHECKING, Any, Literal, TypedDict, overload

from typing_extensions import Unpack, override

from mqns.entity.entity import Entity
from mqns.entity.memory.event import (
    MemoryReadRequestEvent,
    MemoryReadResponseEvent,
    MemoryWriteRequestEvent,
    MemoryWriteResponseEvent,
)
from mqns.entity.memory.memory_qubit import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import QNode
from mqns.models.core import QuantumModel, QuantumModelT
from mqns.models.delay import DelayInput, parseDelay
from mqns.models.epr import BaseEntanglement
from mqns.simulator import Event, func_to_event
from mqns.simulator.simulator import Simulator

if TYPE_CHECKING:
    from mqns.entity.qchannel import QuantumChannel


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

        self._by_qchannel = dict["QuantumChannel", list[int]]()
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
        simulator = self.simulator

        if isinstance(event, MemoryReadRequestEvent):
            result = self.read(event.key)  # will not update fidelity
            simulator.add_event(
                MemoryReadResponseEvent(self.node, result, request=event, t=simulator.tc + self.delay.calculate(), by=self)
            )
        elif isinstance(event, MemoryWriteRequestEvent):
            result = self.write(event.qubit)
            simulator.add_event(
                MemoryWriteResponseEvent(self.node, result, request=event, t=simulator.tc + self.delay.calculate(), by=self)
            )

    @property
    def count(self) -> int:
        """Return the quantity of stored qubits."""
        return self._usage

    @overload
    def find(
        self,
        predicate: Callable[[MemoryQubit, QuantumModel | None], bool],
        *,
        qchannel: "QuantumChannel|None" = None,
    ) -> Iterator[tuple[MemoryQubit, QuantumModel | None]]:
        pass

    @overload
    def find(
        self,
        predicate: Callable[[MemoryQubit, BaseEntanglement], bool],
        *,
        has_epr: Literal[True],
        qchannel: "QuantumChannel|None" = None,
    ) -> Iterator[tuple[MemoryQubit, BaseEntanglement]]:
        pass

    def find(
        self,
        predicate: Callable[[MemoryQubit, Any], bool],
        *,
        has_epr=False,
        qchannel: "QuantumChannel|None" = None,
    ) -> Iterator[Any]:
        """
        Iterate over qubits and associated data that satisfy a predicate.

        Args:
            predicate: Callback function to accept or reject each qubit and associated data.
            qchannel: If set, only qubits assigned to specified quantum channel are considered.
            has_epr: If true, only qubits with associated entanglements are considered.
        """
        # This function is a hot spot for performance optimization.
        # - self._by_qchannel speeds up filtering by assigned qchannel
        # - hasattr(qm, "ch_index") is faster than isinstance(qm, BaseEntanglement)
        iterable: Iterable[tuple[MemoryQubit, QuantumModel | None]] = self._storage
        if qchannel is not None:
            ch_addrs = self._by_qchannel.get(qchannel, [])
            iterable = (self._storage[addr] for addr in ch_addrs)
        for qubit, qm in iterable:
            if (not has_epr or hasattr(qm, "ch_index")) and predicate(qubit, qm):
                yield (qubit, qm)

    def assign(self, ch: "QuantumChannel", *, n=1) -> list[int]:
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

    def allocate(self, ch: "QuantumChannel", path_id: int, path_direction: PathDirection, *, n=1) -> list[int]:
        """
        Allocate n qubits to a given path ID.

        This method searches for n qubits that have not yet been allocated
        (i.e., `path_id` is None), assigns them the provided `path_id`, and returns their addresses.

        Args:
            ch: The quantum channel to which the memory qubit has been assigned.
            path_id: The identifier of the entanglement path to which the memory qubit will be allocated.
            path_direction: The end of the path to which the qubit allocated qubit points.
                    This is typically either the source or the destination of the path (LEFT/RIGHT).

        Returns:
            List of qubit addresses.

        Raises:
            OverflowError - insufficient unallocated qubits.
        """
        addrs: list[int] = []
        for qubit, _ in itertools.islice(self.find(lambda q, _: q.path_id is None, qchannel=ch), n):
            qubit.path_id = path_id
            qubit.path_direction = path_direction
            addrs.append(qubit.addr)

        if len(addrs) != n:
            raise OverflowError(f"{self}: insufficient qubits for allocate(n={n})")
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

    def _find_by_key(self, key: int | str) -> int:
        if type(key) is int:
            return key if key < self.capacity else -1

        if type(key) is str:
            for qubit, _ in self.find(lambda _, v: getattr(v, "name", None) == key):
                return qubit.addr

        return -1

    @overload
    def read(self, key: int | str, *, must: None = None, remove=False) -> tuple[MemoryQubit, QuantumModel | None] | None:
        """
        Retrieve a qubit and associated quantum model.

        Args:
            key: Qubit address or EPR name.
            must: None.
            remove: Whether to remove the quantum model.

        Returns:
            Qubit and associated quantum model (possibly empty), or None if it does not exist.
        """
        pass

    @overload
    def read(self, key: int | str, *, must: Literal[True], remove=False) -> tuple[MemoryQubit, QuantumModel | None]:
        """
        Retrieve a qubit and associated quantum model.

        Args:
            key: Qubit address or EPR name.
            must: True.
            remove: Whether to remove the quantum model.

        Returns:
            Qubit and associated quantum model (possibly empty).

        Raises:
            IndexError - qubit not found.
        """
        pass

    @overload
    def read(
        self, key: int | str, *, must: type[QuantumModelT], set_fidelity=False, remove=False
    ) -> tuple[MemoryQubit, QuantumModelT]:
        """
        Retrieve a qubit and associated quantum model.

        Args:
            key: Qubit address or EPR name.
            must: Expected subclass of QuantumModel, tested with `type(data) is expected_type`.
            set_fidelity: Whether to update fidelity, XXX current formula is inaccurate for EPRs.
            remove: Whether to remove the quantum model.

        Returns:
            Qubit and associated quantum model (has type specified in `must`).

        Raises:
            IndexError - qubit not found.
            ValueError - no quantum information is stored or it is not the expected type.
        """
        pass

    def read(self, key: int | str, *, must: bool | type[QuantumModelT] | None = None, set_fidelity=False, remove=False):
        addr = self._find_by_key(key)
        if addr == -1:
            if must:
                raise IndexError(f"{self}: cannot find {key}")
            return None

        qubit, data = self._storage[addr]
        if must not in (None, True) and type(data) is not must:
            raise ValueError(f"{self}: data at {addr} is not {must}")

        if set_fidelity and isinstance(data, BaseEntanglement) and not data.read:
            data.read = True
            now = self.simulator.tc
            data.store_error_model((now - data.creation_time).sec, self.decoherence_rate)

        if remove:
            self._usage -= 1
            self._storage[addr] = (qubit, None)

            # cancel scheduled decoherence event
            qubit.set_event(QuantumMemory, None)

        return qubit, data

    def write(
        self, qm: QuantumModel, *, path_id: int | None = None, address: int | None = None, key: str | None = None
    ) -> MemoryQubit | None:
        """
        Store a quantum model (e.g., a qubit or an entangled pair) in memory.

        1. Find an unused memory slot that satisfies optional constraints.
        2. Store and provided quantum model.
        3. Schedule a decoherence event based on the memory's decoherence rate.

        Args:
            qm: The quantum model to store (e.g., an entangled pair).
            path_id: Constrain by `MemoryQubit.path_id` field (set by `allocate`).
            address: Constrain by qubit address.
            key: Constrain by `MemoryQubit.active` field (set in `LinkLayer` reservation).

        Returns:
            Qubit where the quantum model was stored, or None if no storage available.
        """

        qubit, _ = next(
            self.find(
                lambda q, v: v is None
                and (path_id is None or q.path_id == path_id)
                and (address is None or q.addr == address)
                and (key is None or key == q.active)
            ),
            (None, None),
        )
        if qubit is None:
            return None

        self._storage[qubit.addr] = (qubit, qm)
        self._usage += 1

        if isinstance(qm, BaseEntanglement):
            qm.decoherence_time = qm.creation_time + self.decoherence_delay
            self._schedule_decohere(qubit, qm)

        return qubit

    def update(self, old_qm: str, new_qm: QuantumModel) -> bool:
        """
        Update the data of a stored qubit without resetting its coherence time.

        This method replaces an existing quantum model in memory with a new one,
        preserving the original decoherence schedule. The old decoherence event is
        canceled and replaced with a new event for the updated model, but at the
        same decoherence time.

        Args:
            old_qm (str): The name of the existing `QuantumModel` to be replaced.
            new_qm (QuantumModel): The new `QuantumModel` instance to store in its place.

        Returns:
            bool: True if the update was successful, False if `old_qm` was not found
                (e.g., already decohered or removed).

        Notes:
            - If `old_qm` is not found in memory but a pending decoherence event still exists,
              that event is canceled and removed to avoid inconsistency.
            - This method does not modify the coherence time or reschedule it—only updates
              the stored quantum model while maintaining timing integrity.
        """
        addr = self._find_by_key(old_qm)
        if addr == -1:
            return False

        qubit = self._storage[addr][0]
        self._storage[addr] = (qubit, new_qm)

        if isinstance(new_qm, BaseEntanglement):
            self._schedule_decohere(qubit, new_qm)
        else:
            qubit.set_event(QuantumMemory, None)

        return True

    def clear(self) -> None:
        """Clear all qubits in the memory."""
        for qubit, _ in self._storage:
            qubit.reset_state()
            self._storage[qubit.addr] = (qubit, None)
        self._usage = 0

    def get_channel_qubits(self, ch: "QuantumChannel") -> list[tuple[MemoryQubit, QuantumModel | None]]:
        """Retrieve all memory qubits associated with a specific quantum channel.

        This method returns all memory qubits that are linked to the given quantum channel name,
        regardless of their state or usage.

        Args:
            ch: The quantum channel to filter qubits by.

        Returns:
            A list of memory qubits (along with their `QuantumModel`) that are bound to the specified quantum channel.
        """
        return list(self.find(lambda q, v: True, qchannel=ch))

    def _schedule_decohere(self, qubit: MemoryQubit, epr: BaseEntanglement):
        simulator = self.simulator
        assert epr.decoherence_time is not None
        assert epr.decoherence_time >= simulator.tc
        event = func_to_event(epr.decoherence_time, self.decohere_qubit, qubit, epr, by=self)
        qubit.set_event(QuantumMemory, event)
        simulator.add_event(event)

    def decohere_qubit(self, qubit: MemoryQubit, qm: BaseEntanglement):
        """
        Mark the `BaseEntanglement` quantum model associated with a qubit as decohered.
        This is invoked through decoherence event from `_schedule_decohere`.

        1. Check whether the EPR pair still exists in memory via `self.read(key=qm)`.

           - If the qubit was already released via swap/purify, it is safely ignored.
           - If the qubit was re-entangled, it would be associated with a different quantum model
             and would not be found via `self.read(key=qm)`.

        2. If the EPR is still present, set the qubit to RELEASE state, and then
           schedule a `QubitDecoheredEvent` to inform the link layer.

        Args:
            qubit: The memory qubit that has reached its decoherence time.
            qm: The assocated quantum model, which must also be an instance of `BaseEntanglement`.
        """
        from mqns.network.protocol.event import QubitDecoheredEvent  # noqa: PLC0415

        simulator = self.simulator

        qm.is_decoherenced = True
        if self.read(qm.name, remove=True) is None:
            return

        qubit.state = QubitState.RELEASE
        simulator.add_event(QubitDecoheredEvent(self.node, qubit, t=simulator.tc, by=self))

    def __repr__(self) -> str:
        return "<memory " + self.name + ">"
