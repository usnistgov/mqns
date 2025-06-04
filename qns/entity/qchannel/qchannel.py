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

from typing import Any

from qns.entity.entity import Entity
from qns.entity.node import QNode
from qns.models.core import QuantumModel
from qns.models.delay import DelayInput, parseDelay
from qns.models.epr import BaseEntanglement
from qns.simulator import Event, Simulator, Time
from qns.utils import get_rand, log


class QuantumChannel(Entity):
    """QuantumChannel is the channel for transmitting qubit
    """

    def __init__(self, name: str|None = None, node_list: list[QNode] = [], *,
                 bandwidth: int = 0, delay: DelayInput = 0, drop_rate: float = 0,
                 max_buffer_size: int = 0, length: float = 0, decoherence_rate: float = 0,
                 transfer_error_model_args: dict = {}):
        """Args:
        name (str): the name of this channel
        node_list (List[QNode]): a list of QNodes that it connects to
        bandwidth (int): the qubit per second on this channel. 0 represents unlimited
        delay (float): the time delay for transmitting a packet, or a ``DelayModel``
        drop_rate (float): probability of photon loss. 0 means never, 1 means always.
        max_buffer_size (int): the max buffer size.
            If it is full, the next coming packet will be dropped. 0 represents unlimited.

        length (float): the length of this channel
        decoherence_rate: the decoherence rate that will pass to the transfer_error_model
        transfer_error_model_args (dict): the parameters that pass to the transfer_error_model

        """
        super().__init__(name=name)
        self.node_list = node_list.copy()
        self.bandwidth = bandwidth
        self.delay_model = parseDelay(delay)
        self.drop_rate = drop_rate
        assert 0.0 <= self.drop_rate <= 1.0
        self.max_buffer_size = max_buffer_size
        self.length = length
        self.decoherence_rate = decoherence_rate
        self.transfer_error_model_args = transfer_error_model_args
        self._next_send_time: Time

    def install(self, simulator: Simulator) -> None:
        """``install`` is called before ``simulator`` runs to initialize or set initial events

        Args:
            simulator (Simulator): the simulator

        """
        super().install(simulator)
        self._next_send_time = self.simulator.ts

    def send(self, qubit: QuantumModel, next_hop: QNode):
        """Send a qubit to the next_hop

        Args:
            qubit (QuantumModel): the transmitting qubit
            next_hop (QNode): the next hop QNode
        Raises:
            NextHopNotConnectionException: the next_hop is not connected to this channel

        """
        simulator = self.simulator

        if next_hop not in self.node_list:
            raise NextHopNotConnectionException

        if self.bandwidth != 0:

            if self._next_send_time <= simulator.current_time:
                send_time = simulator.current_time
            else:
                send_time = self._next_send_time

            if self.max_buffer_size != 0 and \
               send_time > simulator.current_time + self.max_buffer_size / self.bandwidth:
                # buffer is overflow
                log.debug(f"qchannel {self}: drop qubit {qubit} due to overflow")
                return

            self._next_send_time = send_time + 1 / self.bandwidth
        else:
            send_time = simulator.current_time

        # random drop
        if self.drop_rate > 0 and get_rand() < self.drop_rate:
            log.debug(f"qchannel {self}: drop qubit {qubit} due to drop rate")
            if isinstance(qubit, BaseEntanglement):
                qubit.set_decoherenced(True) # photon is lost -> flag this pair as decoherenced to inform receiver node
            return

        # add delay
        recv_time = send_time + self.delay_model.calculate()

        # operation on the qubit
        qubit.transfer_error_model(self.length, self.decoherence_rate, **self.transfer_error_model_args)
        send_event = RecvQubitPacket(recv_time, name=None, by=self, qchannel=self,
                                     qubit=qubit, dest=next_hop)
        simulator.add_event(send_event)

    def __repr__(self) -> str:
        if self.name is not None:
            return "<qchannel "+self.name+">"
        return super().__repr__()


class NextHopNotConnectionException(Exception):
    pass


class RecvQubitPacket(Event):
    """The event for a QNode to receive a classic packet
    """

    def __init__(self, t: Time|None = None, qchannel: QuantumChannel|None = None,
                 qubit: QuantumModel|None = None, dest: QNode|None = None, name: str|None = None, by: Any = None):
        super().__init__(t=t, name=name, by=by)
        self.qchannel = qchannel
        self.qubit = qubit
        self.dest = dest

    def invoke(self) -> None:
        assert self.dest is not None
        self.dest.handle(self)
