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

import itertools
from collections.abc import Callable, Iterator
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
from mqns.models.core import QuantumModel
from mqns.models.delay import DelayInput, parseDelay
from mqns.models.epr import BaseEntanglement
from mqns.simulator import Event, func_to_event

if TYPE_CHECKING:
    from mqns.entity.qchannel import QuantumChannel


class QuantumMemoryInitKwargs(TypedDict, total=False):
    capacity: int
    delay: DelayInput
    decoherence_rate: float
    store_error_model_args: dict


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
            delay: async read/write delay in seconds, or a ``DelayModel``
            decoherence_rate: decoherence rate passed to `QuantumModel.store_error_model()`.
                              0 means the memory will never lose coherence.
            store_error_model_args: parameters passed to `QuantumModel.store_error_model()`.
        """
        super().__init__(name=name)
        self.node: QNode | None = None
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
        self.decoherence_rate = kwargs.get("decoherence_rate", 0.0)
        self.store_error_model_args = kwargs.get("store_error_model_args", {})

        assert self.decoherence_rate >= 0.0
        self.decoherence_delay = None if self.decoherence_rate == 0.0 else 1 / self.decoherence_rate
        """
        Inverse of `decoherence_rate`, i.e. duration between EPR creation time and decoherence time in seconds.
        """

        assert self.capacity >= 1
        self._storage: list[tuple[MemoryQubit, QuantumModel | None]] = [
            (MemoryQubit(addr), None) for addr in range(self.capacity)
        ]
        self._usage = 0

        self.pending_decohere_events: dict[str, Event] = {}
        """
        Table of pending decoherence events.
        Key is `BaseEntanglement.name`.
        Value is the decohere event, which should be uncanceled and unexpired.
        """

    @override
    def handle(self, event: Event) -> None:
        assert self.node is not None
        simulator = self.simulator

        if isinstance(event, MemoryReadRequestEvent):
            result = self.read(event.key)
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

    def is_full(self) -> bool:
        """
        Check whether the memory is full
        """
        return self._usage >= self.capacity

    @overload
    def find(
        self, predicate: Callable[[MemoryQubit, QuantumModel | None], bool]
    ) -> Iterator[tuple[MemoryQubit, QuantumModel | None]]:
        """
        Iterable over qubits and associated data that satisfy a predicate.
        """
        pass

    @overload
    def find(
        self, predicate: Callable[[MemoryQubit, BaseEntanglement], bool], *, has_epr: Literal[True]
    ) -> Iterator[tuple[MemoryQubit, BaseEntanglement]]:
        """
        Iterable over qubits and associated data that satisfy a predicate.
        Only qubits with associated entanglements are considered.
        """
        pass

    def find(self, predicate: Callable[[MemoryQubit, Any], bool], *, has_epr=False) -> Iterator[Any]:
        for qubit, qm in self._storage:
            if has_epr and not isinstance(qm, BaseEntanglement):
                continue
            if predicate(qubit, qm):
                yield (qubit, qm)

    def assign(self, ch: "QuantumChannel", n=1) -> list[int]:
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
            qubit.assign(ch)
            addrs.append(qubit.addr)
        if len(addrs) != n:
            raise OverflowError(f"{self}: insufficient qubits for assign(n={n})")
        return addrs

    def unassign(self, *addrs: int) -> None:
        """
        Unassign one or more qubits from any quantum channel.
        """
        for addr in addrs:
            self._storage[addr][0].unassign()

    def allocate(self, path_id: int, path_direction: PathDirection, *, ch_name: str, n=1) -> list[int]:
        """
        Allocate n qubits to a given path ID.

        This method searches for n qubits that have not yet been allocated
        (i.e., `path_id` is None), assigns them the provided `path_id`, and returns their addresses.

        Args:
            path_id: The identifier of the entanglement path to which the memory qubit will be allocated.
            ch_name: The name of the quantum channel to which the memory qubit is assigned.
            path_direction: The end of the path to which the qubit allocated qubit points.
                    This is typically either the source or the destination of the path (LEFT/RIGHT).

        Returns:
            List of qubit addresses.

        Raises:
            OverflowError - insufficient unallocated qubits.
        """
        addrs: list[int] = []
        for qubit, _ in itertools.islice(
            self.find(lambda q, _: (q.path_id is None and (q.qchannel is not None and q.qchannel.name == ch_name))),
            n,
        ):
            qubit.allocate(path_id, path_direction)
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
            self._storage[addr][0].deallocate()

    def _find_by_key(self, key: int | QuantumModel | str) -> int:
        if isinstance(key, int):
            return key if key < self.capacity else -1

        if isinstance(key, QuantumModel):
            for qubit, _ in self.find(lambda _, v: v == key):
                return qubit.addr

        if isinstance(key, str):
            for qubit, _ in self.find(lambda _, v: getattr(v, "name", None) == key):
                return qubit.addr

        return -1

    @overload
    def get(self, key: int | QuantumModel | str, *, must: None = None) -> tuple[MemoryQubit, QuantumModel | None] | None:
        """
        Retrieve a qubit and associated quantum model without removing it.

        Args:
            key: Qubit address or quantum model or EPR name.

        Returns:
            Qubit and associated quantum model, or None if it does not exist.
        """
        pass

    @overload
    def get(self, key: int | QuantumModel | str, *, must: Literal[True]) -> tuple[MemoryQubit, QuantumModel | None]:
        """
        Retrieve a qubit and associated quantum model without removing it.

        Args:
            key: Qubit address or quantum model or EPR name.
            must: True.

        Returns:
            Qubit and associated quantum model.

        Raises:
            IndexError - qubit not found.
        """
        pass

    def get(self, key: int | QuantumModel | str, *, must: bool | None = None) -> tuple[MemoryQubit, QuantumModel | None] | None:
        idx = self._find_by_key(key)
        if idx != -1:
            return self._storage[idx]
        elif must:
            raise IndexError(f"{self}: cannot find {key}")
        else:
            return None

    @overload
    def read(
        self, key: int | QuantumModel | str, *, destructive=True, must: None = None
    ) -> tuple[MemoryQubit, QuantumModel] | None:
        """
        Read a qubit and set fidelity on associated quantum model.

        Args:
            key: Qubit address or quantum model or EPR name.
            destructive: If True, remove the quantum model after reading.

        Returns:
            Qubit and associated quantum model, or None if it does not exist.
        """
        pass

    @overload
    def read(self, key: int | QuantumModel | str, *, destructive=True, must: Literal[True]) -> tuple[MemoryQubit, QuantumModel]:
        """
        Read a qubit and set fidelity on associated quantum model.

        Args:
            key: Qubit address or quantum model or EPR name.
            destructive: If True, remove the quantum model after reading.
            must: True.

        Returns:
            Qubit and associated quantum model.

        Raises:
            IndexError - qubit not found.
            ValueError - no quantum information is stored
        """
        pass

    def read(
        self, key: int | QuantumModel | str, *, destructive=True, must: bool | None = None
    ) -> tuple[MemoryQubit, QuantumModel] | None:
        addr = self._find_by_key(key)
        if addr == -1:
            if must:
                raise IndexError(f"{self}: cannot find {key}")
            return None

        (qubit, data) = self._storage[addr]
        if not data:
            if must:
                raise ValueError(f"{self}: no data at index {addr}")
            return None

        if destructive:
            self._usage -= 1
            self._storage[addr] = (qubit, None)

        if isinstance(data, BaseEntanglement):
            self._read_epr(data, destructive)

        return (qubit, data)

    def _read_epr(self, data: QuantumModel, destructive: bool):
        assert isinstance(data, BaseEntanglement)
        assert data.creation_time is not None

        sec_diff = self.simulator.tc.sec - data.creation_time.sec

        # set fidelity at read time
        if not data.read:
            data.store_error_model(t=sec_diff, decoherence_rate=self.decoherence_rate, **self.store_error_model_args)
            data.read = True

        if destructive:
            # cancel scheduled decoherence event
            event = self.pending_decohere_events.pop(data.name, None)
            if event is not None:
                event.cancel()

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

        if isinstance(qm, BaseEntanglement) and qm.creation_time is not None and self.decoherence_delay:
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
            old_event = self.pending_decohere_events.pop(old_qm, None)
            assert old_event is None, f"decohere event not cleared for {old_qm}"
            return False

        qubit = self._storage[addr][0]
        self._storage[addr] = (qubit, new_qm)

        old_event = self.pending_decohere_events.pop(old_qm)
        old_event.cancel()

        if isinstance(new_qm, BaseEntanglement):
            self._schedule_decohere(qubit, new_qm)

        return True

    def clear(self) -> None:
        """Clear all qubits in the memory."""
        for qubit, _ in self._storage:
            qubit.reset_state()
            self._storage[qubit.addr] = (qubit, None)
        self._usage = 0

        for event in self.pending_decohere_events.values():
            event.cancel()
        self.pending_decohere_events.clear()

    def get_channel_qubits(self, ch_name: str) -> list[tuple[MemoryQubit, QuantumModel | None]]:
        """Retrieve all memory qubits associated with a specific quantum channel.

        This method returns all memory qubits that are linked to the given quantum channel name,
        regardless of their state or usage.

        Args:
            ch_name (str): The name of the quantum channel to filter qubits by.

        Returns:
            List[Tuple[MemoryQubit, QuantumModel]]: A list of memory qubits (along with their `QuantumModel`)
            that are bound to the specified quantum channel.

        """
        return list(self.find(lambda q, _: q.qchannel is not None and q.qchannel.name == ch_name))

    def _schedule_decohere(self, qubit: MemoryQubit, epr: BaseEntanglement):
        simulator = self.simulator
        assert epr.decoherence_time is not None
        assert epr.decoherence_time >= simulator.tc
        event = func_to_event(epr.decoherence_time, self.decohere_qubit, qubit, epr, by=self)
        self.pending_decohere_events[epr.name] = event
        simulator.add_event(event)

    def decohere_qubit(self, qubit: MemoryQubit, qm: QuantumModel):
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

        assert self.node is not None
        simulator = self.simulator
        assert isinstance(qm, BaseEntanglement)

        qm.is_decoherenced = True
        if self.read(qm) is None:
            return

        qubit.state = QubitState.RELEASE
        simulator.add_event(QubitDecoheredEvent(self.node, qubit, t=simulator.tc, by=self))

    def __repr__(self) -> str:
        return "<memory " + self.name + ">"
