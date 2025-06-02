#    Modified by Amar Abane for Multiverse Quantum Network Simulator
#    Date: 05/17/2025
#    Summary of changes: Adapted logic to support dynamic approaches.
#
#    This file is based on a snapshot of SimQN (https://github.com/qnslab/SimQN),
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


from typing import TYPE_CHECKING

from qns.entity.entity import Entity
from qns.entity.memory.event import (
    MemoryReadRequestEvent,
    MemoryReadResponseEvent,
    MemoryWriteRequestEvent,
    MemoryWriteResponseEvent,
)
from qns.entity.memory.memory_qubit import MemoryQubit, QubitState
from qns.entity.node import QNode
from qns.models.core import QuantumModel
from qns.models.delay import DelayInput, parseDelay
from qns.models.epr import BaseEntanglement
from qns.simulator import Event, Simulator, Time, func_to_event
from qns.utils import log

if TYPE_CHECKING:
    from qns.entity.qchannel import QuantumChannel
    from qns.network.protocol.link_layer import LinkLayer


class QuantumMemory(Entity):
    """Quantum memory stores qubits or entangled pairs

    It has two modes:
        Synchronous mode, users can use the ``read`` and ``write`` function to operate the memory directly without delay
        Asynchronous mode, users can use events to operate memories asynchronously
    """

    def __init__(self, name: str|None = None, node: QNode|None = None, *,
                 capacity: int = 1, delay: DelayInput = 0,
                 decoherence_rate: float = 0, store_error_model_args: dict = {}):
        """Args:
        name (str): memory name
        node (QNode): the quantum node that equips this memory
        capacity (int): the capacity of this quantum memory. 0 represents unlimited.
        delay (Union[float,DelayModel]): the read and write delay in second, or a ``DelayModel``
        decoherence_rate (float): the decoherence rate of this memory that will pass to the store_error_model.
                                  0 means the memory will never lose coherence.
        store_error_model_args (dict): the parameters that will pass to the store_error_model

        """
        super().__init__(name=name)
        self.node = node
        self.capacity = capacity
        self.delay_model = parseDelay(delay)

        if self.capacity > 0:
            self._storage: list[tuple[MemoryQubit, QuantumModel|None]] = [
                (MemoryQubit(addr), None) for addr in range(self.capacity)]
        else:      # should not use this case
            raise ValueError("Error: unlimited memory capacity not supported")

        self._usage = 0
        self.decoherence_rate = decoherence_rate
        self.store_error_model_args = store_error_model_args

        self.link_layer: "LinkLayer|None" = None

        self.pending_decohere_events: dict[str, Event] = {}
        """map of future qubit decoherence events"""


    def install(self, simulator: Simulator) -> None:
        super().install(simulator)
        assert self.node is not None

        from qns.network.protocol.link_layer import LinkLayer
        ll_apps = self.node.get_apps(LinkLayer)
        try:
            self.link_layer = ll_apps[0]
        except IndexError:
            pass


    def _search(self, key: QuantumModel|str|None = None, address: int|None = None) -> int:
        """This method searches through the internal storage for a matching qubit based on either
        its memory address or a key related to the stored EPR.

        Args:
            key (Optional[Union[QuantumModel, str, int]]): Identifier for the qubit.
                - If a `QuantumModel`, it matches against the stored EPR instances.
                - If a str, it matches against the stored EPR names.
            address (Optional[int]): The memory address of the qubit to locate.

        Returns:
            int: The index of the matching qubit in the storage list.
                Returns -1 if no matching entry is found.

        Notes:
            - Address, when provided, has precedence over key if both are provided.

        """
        if address is not None:
            for idx, (qubit, _) in enumerate(self._storage):
                if qubit.addr == address:
                    return idx
        elif isinstance(key, QuantumModel):
            for idx, (_, data) in enumerate(self._storage):
                if data == key:
                    return idx
        elif isinstance(key, str):
            for idx, (_, data) in enumerate(self._storage):
                if getattr(data, "name", None) == key:
                    return idx
        return -1

    @property
    def count(self) -> int:
        """Return the quantity of stored qubits."""
        return self._usage

    def get(self, key: QuantumModel|str|None = None, address: int|None = None) -> tuple[MemoryQubit, QuantumModel|None]|None:
        """
        Retrieve a qubit from memory without removing it.

        Args:
            key (Optional[Union[QuantumModel, str, int]]): Identifier for the qubit.
                - If a `QuantumModel`, it matches against the stored EPR instances.
                - If a str, it matches against the stored EPR names.
            address (Optional[int]): The memory address of the qubit to locate.

        Returns:
            Tuple[MemoryQubit, Optional[QuantumModel]]:
                - The `MemoryQubit` metadata object of the specified location.
                - The associated `QuantumModel`.
                - None if qubit not found.
        """

        idx = self._search(key=key, address=address)
        if idx != -1:
            return self._storage[idx]
        else:
            return None

    def read(self, key: QuantumModel|str|None = None, address: int|None = None,
             destructive = True) -> tuple[MemoryQubit, QuantumModel]|None:
        """
        Reading of a qubit from the memory. This methods sets the fidelity of the EPR at read time.

        Args:
            key (Optional[Union[QuantumModel, str, int]]): Identifier for the qubit.
                - If a `QuantumModel`, it matches against the stored EPR instances.
                - If a str, it matches against the stored EPR names.
            address (Optional[int]): The memory address of the qubit to locate.
            destructive: (bool): Whether to delete the EPR after reading (default True).

        Returns:
            Tuple[MemoryQubit, Optional[QuantumModel]]:
                - The `MemoryQubit` metadata objeect of the specified location.
                - The associated `QuantumModel`.
                - None if qubit not found or no quantum information is stored.
        """
        assert self._simulator is not None

        idx = self._search(key=key, address=address)
        if idx == -1:
            return None

        (qubit, data) = self._storage[idx]
        if not data:
            return None

        if destructive:
            self._usage -= 1
            self._storage[idx] = (qubit, None)

        if isinstance(data, BaseEntanglement):
            self._read_epr(data, destructive)

        return (qubit, data)

    def _read_epr(self, data: QuantumModel, destructive: bool):
        assert self._simulator is not None
        assert isinstance(data, BaseEntanglement)
        assert data.creation_time is not None
        assert data.name is not None

        t_now = self._simulator.current_time
        sec_diff = t_now.sec - data.creation_time.sec

        # set fidelity at read time
        if not data.read:
            data.store_error_model(t=sec_diff, decoherence_rate=self.decoherence_rate, **self.store_error_model_args)
            data.read = True

        if destructive:
            # cancel scheduled decoherence event
            if data.name in self.pending_decohere_events:
                event = self.pending_decohere_events[data.name]
                event.cancel()
                self.pending_decohere_events.pop(data.name)

    def write(self, qm: QuantumModel, path_id: int|None = None,
              address: int|None = None, key: str|None = None) -> MemoryQubit|None:
        """
        Store a quantum model (e.g., a qubit or an entangled pair) in memory.

        This method finds an available memory slot that satisfies optional constraints and
        stores the provided `QuantumModel`. If successful, it schedules a decoherence event
        based on the memory's decoherence rate and returns the corresponding `MemoryQubit`.

        Args:
            qm (QuantumModel): The quantum model to store (e.g., an entangled pair).
            path_id (Optional[int]): Optional path ID to match against the memory qubit.
            address (Optional[int]): Optional qubit address to match.
            key (str, optional): Optional tag to match against the `active` field of
              the memory qubit (obtained from reservation).

        Returns:
            Optional[MemoryQubit]: The `MemoryQubit` object into which the quantum model was stored,
                or `None` if storage was unsuccessful (e.g., memory full or no matching slot found).

        Notes:
            - The decoherence time is calculated as 1 / `decoherence_rate` after the adjusted current time.
        """

        if self.is_full():
            return None

        idx = -1
        for i, (q, v) in enumerate(self._storage):
            if v is None and (key is None or key == q.active):           # Check if the slot is empty
                if (path_id is None or q.path_id == path_id) and (address is None or q.addr == address):
                    idx = i
                    break

        if idx == -1:
            return None

        qubit = self._storage[idx][0]
        self._storage[idx] = (qubit, qm)
        self._usage += 1

        if not isinstance(qm, BaseEntanglement):
            return qubit
        assert self._simulator is not None
        assert qm.creation_time is not None
        assert qm.name is not None

        # schedule an event at T_coh to decohere the qubit
        if self.decoherence_rate:
            decoherence_t = qm.creation_time + Time(sec = 1 / self.decoherence_rate)
            event = func_to_event(decoherence_t, self.decohere_qubit, by=self, qubit=qubit, qm=qm)
            self.pending_decohere_events[qm.name] = event
            self._simulator.add_event(event)
            qm.decoherence_time = decoherence_t

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
        idx = self._search(key=old_qm)
        if idx == -1:
            if old_qm in self.pending_decohere_events:
                print("UNEXPECTED ==> decohere event not cleared")
                old_event = self.pending_decohere_events[old_qm]
                old_event.cancel()
                self.pending_decohere_events.pop(old_qm)
            return False

        qubit = self._storage[idx][0]
        self._storage[idx] = (qubit, new_qm)

        old_event = self.pending_decohere_events[old_qm]
        old_event.cancel()
        self.pending_decohere_events.pop(old_qm)

        if not isinstance(new_qm, BaseEntanglement):
            return True
        assert self._simulator is not None
        assert new_qm.decoherence_time is not None
        assert new_qm.name is not None

        # schedule an event at old T_coh to decohere the qubit
        new_event = func_to_event(new_qm.decoherence_time, self.decohere_qubit, by=self, qubit=self._storage[idx][0], qm=new_qm)
        self.pending_decohere_events[new_qm.name] = new_event
        self._simulator.add_event(new_event)
        return True

    def clear(self) -> None:
        """Clear all qubits in the memory
        """
        for idx, (qubit, _) in enumerate(self._storage):
            qubit.fsm.to_release()
            self._storage[idx] = (qubit, None)
        self._usage = 0
        for _, event in self.pending_decohere_events.items():
            event.cancel()
        self.pending_decohere_events = {}

    def allocate(self, path_id: int) -> int:
        """
        Allocate an unused memory qubit to a given path ID.

        This method searches for the first available memory qubit that has not yet been allocated
        (i.e., `path_id` is None), assigns it the provided `path_id`, and returns its address.

        Args:
            path_id (int): The identifier of the entanglement path
                       to which the memory qubit will be assigned.

        Returns:
            int: The address of the allocated memory qubit, or -1 if no unallocated qubit is available.

        """
        for (qubit,_) in self._storage:
            if qubit.path_id is None:
                qubit.allocate(path_id)
                return qubit.addr
        return -1

    def deallocate(self, address: int) -> bool:
        """
        Deallocate a memory qubit from any assigned path.

        This method finds the memory qubit with the given address and clears its
        path assignment (i.e., resets its `path_id` to None). It does not modify the
        quantum state or remove the qubit from memory.

        Args:
            address (int): The address of the memory qubit to be deallocated.

        Returns:
            bool: True if the qubit was found and deallocated successfully,
                False if no qubit with the specified address exists.
        """
        for (qubit,_) in self._storage:
            if qubit.addr == address:
                qubit.deallocate()
                return True
        return False

    def search_available_qubits(self, path_id: int | None = None) -> list[MemoryQubit]:
        """Search for available (unoccupied and inactive) memory qubits, optionally filtered by path ID.

        This method returns all memory qubits that:
            - Are currently unoccupied (i.e., no associated `QuantumModel`),
            - Are not active (i.e., not in use/reserved for operations),
            - (Optionally) are assigned to the specified path ID.

        Args:
            path_id (Optional[int]): The path ID to filter qubits by. If None, any path ID is accepted.

        Returns:
            List[MemoryQubit]: A list of available memory qubits satisfying the criteria.
                           Returns an empty list if no such qubits are found.

        """
        qubits = []
        for qubit, data in self._storage:
            if data is not None:
                continue
            if qubit.active:
                continue
            if path_id is not None and qubit.path_id != path_id:
                continue
            qubits.append(qubit)
        return qubits

    def search_eligible_qubits(self, exc_qchannel: str | None = None,
                               path_id: int | None = None) -> list[tuple[MemoryQubit, QuantumModel]]:
        """Search for memory qubits that are eligible for use.

        This method scans the memory for qubits that:
            - Are marked as `ELIGIBLE`,
            - (Optionally) belong to the specified path ID (`path_id`),
            - (Optionally) are not associated with the specified quantum channel (`qchannel`).

        Args:
            exc_qchannel (Optional[str]): The name of the quantum channel to exclude. If None, no exclusion is applied.
            path_id (Optional[int]): The path ID the qubit must be assigned to. If None, any path ID is accepted.

        Returns:
            List[Tuple[MemoryQubit, QuantumModel]]:
                A list of tuples containing eligible memory qubits and their associated `QuantumModel` instances.
                The list is empty if no matching qubits are found.

        """
        qubits = []
        for qubit, data in self._storage:
            if data is None:
                continue
            if qubit.fsm.state != QubitState.ELIGIBLE:
                continue
            if path_id is not None and qubit.path_id != path_id:
                continue
            if exc_qchannel is not None and (qubit.qchannel is None or qubit.qchannel.name == exc_qchannel):
                continue
            qubits.append((qubit, data))
        return qubits

    def search_purif_qubits(self, exc_address: int, partner: str,
                            qchannel: str, path_id: int | None = None,
                            purif_rounds: int = 0) -> list[tuple[MemoryQubit, QuantumModel]]:
        """Search for memory qubits eligible for purification with a given qubit.
        Assumes recurrence purification; i.e., input pairs must have undergone the same number of rounds.

        This method searches for qubits in the `PURIF` state that:
            - Are not the current qubit (by address),
            - Belong to the same quantum channel,
            - Are entangled with the specified partner node (either as source or destination),
            - Have completed the specified number of purification rounds,
            - (Optionally) belong to the same path ID if `path_id` is specified.
              This enforces recurrence purification protocols, which require
        both qubits to have undergone the same number of purification rounds.

        Args:
            exc_address (int): The address of the qubit to exclude.
            partner (str): The name of the entanglement partner node (as `src.name` or `dst.name`).
            qchannel (str): The name of the quantum channel used by the current qubit for entanglement.
            path_id (Optional[int]): The path ID that eligible qubits must match. If None, any path ID is accepted.
            purif_rounds (int): The number of purification rounds the eligible qubit must have undergone.

        Returns:
            List[Tuple[MemoryQubit, QuantumModel]]:
                A list of eligible qubits (along with their associated quantum models) that match the criteria.

        """
        qubits = []
        for qubit, data in self._storage:
            if qubit.addr == exc_address:
                continue
            if data is None:
                continue
            if qubit.fsm.state != QubitState.PURIF:
                continue
            if path_id is not None and qubit.path_id != path_id:
                continue
            if qubit.qchannel is None or qubit.qchannel.name != qchannel:
                continue
            if not isinstance(data, BaseEntanglement):
                continue
            if data.src is None or data.dst is None or (data.src.name != partner and data.dst.name != partner):
                continue
            if qubit.purif_rounds != purif_rounds:
                continue
            qubits.append((qubit, data))
        return qubits

    def get_channel_qubits(self, ch_name: str) -> list[tuple[MemoryQubit, QuantumModel|None]]:
        """Retrieve all memory qubits associated with a specific quantum channel.

        This method returns all memory qubits that are linked to the given quantum channel name,
        regardless of their state or usage.

        Args:
            ch_name (str): The name of the quantum channel to filter qubits by.

        Returns:
            List[Tuple[MemoryQubit, QuantumModel]]: A list of memory qubits (along with their `QuantumModel`)
            that are bound to the specified quantum channel.

        """
        qubits: list[tuple[MemoryQubit, QuantumModel|None]] = []
        for (qubit, data) in self._storage:
            if qubit.qchannel and qubit.qchannel.name == ch_name:
                qubits.append((qubit, data))
        return qubits


    def decohere_qubit(self, qubit: MemoryQubit, qm: QuantumModel):
        """This method is called when a qubit reaches the end of its coherence time.
        It marks the associated `QuantumModel` as decohered and performs the following:

        - Checks whether the EPR pair still exists in memory via `self.read(key=qm)`.
            This ensures that the qubit hasn't already been released (e.g., due to swap or purification).
        - If the EPR is still present:
            - Marks the `MemoryQubit` to `RELEASE`.
            - Creates and schedules a `QubitDecoheredEvent` to inform the link layer of the event.

        Args:
            qubit (MemoryQubit): The memory qubit that has decohered.
            qm (QuantumModel): The quantum model associated with the qubit.

        Notes:
            - If the qubit was already released (e.g., swap, purify),
            this method safely ignores it by failing the `self.read(key=qm)` check.
            - If the qubit was re-entangled, the `read()` will not find the original EPR,
            so no event is raised again.

        """
        assert isinstance(qm, BaseEntanglement)
        qm.is_decoherenced = True
        if self.read(key=qm):
            log.debug(f"{self.node}: EPR decohered -> {qm.name} {qm.src}-{qm.dst}")
            qubit.fsm.to_release()
            if self.link_layer is not None:
                self._emit_decohered_event(qubit)

    def _emit_decohered_event(self, qubit: MemoryQubit):
        assert self._simulator is not None
        assert self.link_layer is not None
        from qns.network.protocol.event import QubitDecoheredEvent

        t = self._simulator.tc # + self._simulator.time(sec=0)
        event = QubitDecoheredEvent(link_layer=self.link_layer, qubit=qubit, t=t, by=self)
        self._simulator.add_event(event)

    def count_unallocated_qubits(self) -> int:
        """Return the number of qubits not allocated to any path ID
        """
        free = self.capacity
        for (qubit,_) in self._storage:
            if qubit.path_id:
                free-=1
        return free

    def is_full(self) -> bool:
        """
        Check whether the memory is full
        """
        return self.capacity > 0 and self._usage >= self.capacity


    # For channel qubit allocation at topology creation time
    def assign(self, ch: "QuantumChannel") -> int:
        """
        Assign a qubit to a particular quantum channel
        """
        for (qubit,_) in self._storage:
            if qubit.qchannel is None:
                qubit.assign(ch)
                return qubit.addr
        return -1

    def unassign(self, address: int) -> bool:
        """
        Unassign a qubit from any quantum channel
        """
        for (qubit,_) in self._storage:
            if qubit.addr == address:
                qubit.unassign()
                return True
        return False


    def __repr__(self) -> str:
        if self.name is not None:
            return "<memory "+self.name+">"
        return super().__repr__()

    def handle(self, event: Event) -> None:
        assert self._simulator is not None
        assert isinstance(self.node, QNode)

        if isinstance(event, MemoryReadRequestEvent):
            key = event.key
            # operate qubits and get measure results
            result = self.read(key)

            t = self._simulator.tc + self._simulator.time(sec=self.delay_model.calculate())
            response = MemoryReadResponseEvent(node=self.node, result=result, request=event, t=t, by=self)
            self._simulator.add_event(response)
        elif isinstance(event, MemoryWriteRequestEvent):
            qubit = event.qubit
            result = self.write(qubit)
            t = self._simulator.tc + self._simulator.time(sec=self.delay_model.calculate())
            response = MemoryWriteResponseEvent(node=self.node, result=result, request=event, t=t, by=self)
            self._simulator.add_event(response)
