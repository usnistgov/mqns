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

from qns.entity.base_channel import BaseChannel, BaseChannelInitKwargs
from qns.entity.node import QNode
from qns.models.core import QuantumModel
from qns.models.epr import BaseEntanglement
from qns.simulator import Event, Time

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


class QuantumChannelInitKwargs(BaseChannelInitKwargs, total=False):
    decoherence_rate: float
    transfer_error_model_args: dict


class QuantumChannel(BaseChannel[QNode]):
    """QuantumChannel is the channel for transmitting qubit"""

    def __init__(self, name: str, **kwargs: Unpack[QuantumChannelInitKwargs]):
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
        super().__init__(name, **kwargs)
        self.decoherence_rate = kwargs.get("decoherence_rate", 0.0)
        self.transfer_error_model_args = kwargs.get("transfer_error_model_args", {})

    def send(self, qubit: QuantumModel, next_hop: QNode):
        """Send a qubit to the next_hop

        Args:
            qubit (QuantumModel): the transmitting qubit
            next_hop (QNode): the next hop QNode
        Raises:
            NextHopNotConnectionException: the next_hop is not connected to this channel

        """
        drop, recv_time = self._send(
            packet_repr=f"qubit {qubit}",
            packet_len=1,
            next_hop=next_hop,
            delay=0,
        )

        if drop:
            # photon is lost -> flag this pair as decoherenced to inform receiver node
            if isinstance(qubit, BaseEntanglement):
                qubit.set_decoherenced(True)
            return

        # operation on the qubit
        qubit.transfer_error_model(self.length, self.decoherence_rate, **self.transfer_error_model_args)
        send_event = RecvQubitPacket(t=recv_time, by=self, qchannel=self, qubit=qubit, dest=next_hop)
        self.simulator.add_event(send_event)

    def __repr__(self) -> str:
        return "<qchannel " + self.name + ">"


class RecvQubitPacket(Event):
    """The event for a QNode to receive a qubit"""

    def __init__(
        self, *, t: Time, name: str | None = None, by: Any = None, qchannel: QuantumChannel, qubit: QuantumModel, dest: QNode
    ):
        super().__init__(t=t, name=name, by=by)
        self.qchannel = qchannel
        self.qubit = qubit
        self.dest = dest

    def invoke(self) -> None:
        self.dest.handle(self)
