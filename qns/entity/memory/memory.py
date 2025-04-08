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

from typing import List, Optional, Union, Tuple
from qns.models.delay.constdelay import ConstantDelayModel
from qns.models.delay.delay import DelayModel
from qns.simulator.simulator import Simulator
from qns.simulator.ts import Time
from qns.models.core.backend import QuantumModel
from qns.entity.entity import Entity
from qns.entity.node.qnode import QNode
from qns.entity.memory.memory_qubit import MemoryQubit, QubitState
from qns.simulator.event import Event, func_to_event
import qns.utils.log as log


class OutOfMemoryException(Exception):
    """
    The exception that the memory is full
    """
    pass


class QuantumMemory(Entity):
    """
    Quantum memory stores qubits or entangled pairs.

    It has two modes:
        Synchronous mode, users can use the ``read`` and ``write`` function to operate the memory directly without delay
        Asynchronous mode, users can use events to operate memories asynchronously
    """
    def __init__(self, name: str = None, node: QNode = None,
                 capacity: int = 0, decoherence_rate: Optional[float] = 0,
                 store_error_model_args: dict = {}, delay: Union[float, DelayModel] = 0):
        """
        Args:
            name (str): its name
            node (QNode): the quantum node that equips this memory
            capacity (int): the capacity of this quantum memory. 0 represents unlimited.
            delay (Union[float,DelayModel]): the read and write delay in second, or a ``DelayModel``
            decoherence_rate (float): the decoherence rate of this memory that will pass to the store_error_model
            store_error_model_args (dict): the parameters that will pass to the store_error_model
        """
        super().__init__(name=name)
        self.node = node
        self.capacity = capacity
        self.delay_model = delay if isinstance(delay, DelayModel) else ConstantDelayModel(delay=delay)

        if self.capacity > 0:
            self._storage: List[Tuple[MemoryQubit, Optional[QuantumModel]]] = [
                    (MemoryQubit(addr), None) for addr in range(self.capacity)
            ]
            self._store_time: List[Optional[Time]] = [None] * self.capacity
        else:      # should not use this case
            print("Error: unlimited memory capacity not supported")
            return

        self._usage = 0
        self.decoherence_rate = decoherence_rate
        self.store_error_model_args = store_error_model_args
        
        self.link_layer = None


    def install(self, simulator: Simulator) -> None:
        from qns.network.protocol.link_layer import LinkLayer
        super().install(simulator)
        ll_apps = self.node.get_apps(LinkLayer)
        if ll_apps:
            self.link_layer = ll_apps[0]
        else:
            raise Exception("No LinkLayer protocol found")

    def _search(self, key: Optional[Union[QuantumModel, str, int]] = None, address: Optional[int] = None) -> int:
        index = -1
        if address is not None:
            for idx, (qubit, _) in enumerate(self._storage):
                if qubit.addr == address:
                    return idx
        elif isinstance(key, int):
            if self.capacity == 0 and key >= 0 and key < self._usage:
                index = key
            elif key >= 0 and key < self.capacity and self._storage[key][1] is not None:
                index = key
        elif isinstance(key, QuantumModel):
            for idx, (_, data) in enumerate(self._storage):
                if data is None:
                    continue
                if data == key:
                    return idx
        elif isinstance(key, str):
            for idx, (_, data) in enumerate(self._storage):
                if data is None:
                    continue
                if data.name == key:
                    return idx
        return index

    def get(self, key: Optional[Union[QuantumModel, str, int]] = None, address: Optional[int] = None) -> Tuple[MemoryQubit, Optional[QuantumModel]]:
        """
        get a qubit from the memory but without removing it from the memory

        Args:
            key (Union[QuantumModel, str, int]): the key. It can be a QuantumModel object,
                its name or the index number.
        """
        idx = self._search(key=key, address=address)
        if idx != -1:
            return self._storage[idx]
        else:
            return None

    def get_store_time(self, key: Optional[Union[QuantumModel, str, int]] = None, address: Optional[int] = None) -> Optional[Time]:
        """
        get the store time of a qubit from the memory

        Args:
            key (Union[QuantumModel, str, int]): the key. It can be a QuantumModel object,
                its name or the index number.
        """
        try:
            idx = self._search(key, address)
            if idx != -1:
                return self._store_time[idx]
            else:
                return None
        except IndexError:
            return None

    def read(self, key: Optional[Union[QuantumModel, str, int]] = None, address: Optional[int] = None) -> Tuple[MemoryQubit, Optional[QuantumModel]]:
        """
        Destructive reading of a qubit from the memory

        Args:
            key (Union[QuantumModel, str]): the key. It can be a QuantumModel object,
                its name or the index number.
        """
        idx = self._search(key=key, address=address)
        if idx == -1:
            return None

        (qubit, data) = self._storage[idx]
        store_time = self._store_time[idx]
        self._usage -= 1

        self._storage[idx] = (self._storage[idx][0], None)
        self._store_time[idx] = None

        t_now = self._simulator.current_time
        sec_diff = t_now.sec - store_time.sec
        data.store_error_model(t=sec_diff, decoherence_rate=self.decoherence_rate, **self.store_error_model_args)
        return (qubit, data)

    def write(self, qm: QuantumModel, pid: Optional[int] = None, address: Optional[int] = None) -> Optional[MemoryQubit]:
        """
        The API for storing a qubit to the memory

        Args:
            qm (QuantumModel): the `QuantumModel`, could be a qubit or an entangled pair

        Returns:
            bool: whether the qubit is stored successfully
        """
        if self.is_full():
            return None

        idx = -1
        for i, (q, v) in enumerate(self._storage):
            if v is None:           # Check if the slot is empty
                if (pid is None or q.pid == pid) and (address is None or q.addr == address):
                    idx = i
                    break
        if idx == -1:
            return None

        self._storage[idx] = (self._storage[idx][0], qm)
        self._store_time[idx] = self._simulator.current_time
        self._usage += 1

        # schedule an event after T_coh to decohere the qubit
        # TODO: use a generation time to coordinate sender-receiver decoherence time 
        t = self._simulator.tc + Time(sec = 1 / self.decoherence_rate)
        event = func_to_event(t, self.decohere_qubit, by=self, qubit=self._storage[idx][0], epr=qm)
        self._simulator.add_event(event)
        
        return self._storage[idx][0]    # return the memory qubit


    def allocate(self, path_id: int) -> int:
        for (qubit,_) in self._storage:
            if qubit.pid is None:
                qubit.allocate(path_id) 
                return qubit.addr
        return -1

    def deallocate(self, address: int) -> bool:
        for (qubit,_) in self._storage:
            if qubit.addr == address:
                qubit.deallocate()    
                return True
        return False
    
    def search_eligible_qubits(self, pid: int = None) -> List[Tuple[MemoryQubit, QuantumModel]]:
        qubits = []
        for (qubit, data) in self._storage:
            if data and qubit.fsm.state == QubitState.ELIGIBLE and qubit.pid == pid:
                qubits.append((qubit, data))
        return qubits

    def is_full(self) -> bool:
        """
        check whether the memory is full
        """
        return self.capacity > 0 and self._usage >= self.capacity

    @property
    def count(self) -> int:
        """
        return the current memory usage
        """
        return self._usage
    
    @property
    def free(self) -> int:
        """
        return the number of non-allocated memory qubits
        """
        free = self.capacity
        for (qubit,_) in self._storage:
            if qubit.pid:
                free-=1
        return free

    def handle(self, event: Event) -> None:
        from qns.entity.memory.event import MemoryReadRequestEvent, MemoryReadResponseEvent, \
                                            MemoryWriteRequestEvent, MemoryWriteResponseEvent
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


    def decohere_qubit(self, qubit: MemoryQubit, epr: QuantumModel):
        # we try to read EPR (not qubit addr) to make sure we are dealing with this particular EPR:
        # - if qubit has been in swap/purify -> L3 should have released qubit and notified L2.
        # - if qubit has been re-entangled, self.read for EPR.name will not find it, so no notification.
        if self.read(key=epr):
            log.debug(f"{self.node}: Qubit {self.name},{qubit} decohered.")
            qubit.fsm.to_release()
            from qns.network.protocol.event import QubitDecoheredEvent
            t = self._simulator.tc + self._simulator.time(sec=0)   # simulate comm. time between Memory and L2
            event = QubitDecoheredEvent(link_layer=self.link_layer, qubit=qubit, t=t, by=self)
            self._simulator.add_event(event)

    def __repr__(self) -> str:
        if self.name is not None:
            return "<memory "+self.name+">"
        return super().__repr__()
