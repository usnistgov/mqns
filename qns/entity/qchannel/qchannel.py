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

from enum import Enum, auto
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


class LinkType(Enum):
    DIM_BK_SEQ = auto()  # detection-in-midpoint with simple Barrett-kok protocol as implemented in Sequence
    DIM_BK = auto()  # detection-in-midpoint with simple Barrett-kok protocol
    SR = auto()  # sender-receiver
    SIM = auto()  # source-in-midpoint
    DIM_BK_TD = auto()  # detection-in-midpoint with simple Barrett-kok protocol and quantum transduction
    SR_TD = auto()  # sender-receiver with quantum transduction
    SIM_TD = auto()  # source-in-midpoint with quantum transduction


class QuantumChannelInitKwargs(BaseChannelInitKwargs, total=False):
    link_architecture: LinkType
    """Type of link architecture for elementary EPR generation"""
    decoherence_rate: float
    """Decoherence rate passed to transfer_error_model."""
    transfer_error_model_args: dict
    """Parameters passed to transfer_error_model."""


class QuantumChannel(BaseChannel[QNode]):
    """QuantumChannel is the channel for transmitting qubit"""

    def __init__(self, name: str, **kwargs: Unpack[QuantumChannelInitKwargs]):
        super().__init__(name, **kwargs)
        self.link_architecture = kwargs.get("link_architecture", LinkType.DIM_BK_SEQ)
        self.decoherence_rate = kwargs.get("decoherence_rate", 0.0)
        self.transfer_error_model_args = kwargs.get("transfer_error_model_args", {})

    def assign_memory_qubits(self, *, capacity: int | dict[str, int] = 1):
        """
        Assign memory qubits at each node connected to the qchannel.

        Args:
            capacity: required quantity of qubits.
                      If given as an integer, this applies to every node.
                      If given as a dict, it should be a mapping from node name to capacity of this node,
                      where every node connected to the qchannel must appear in the dict.

        Raises:
            RuntimeError - insufficient qubits.
        """
        for node in self.node_list:
            memory = node.get_memory()
            cap = capacity if isinstance(capacity, int) else capacity[node.name]
            for _ in range(cap):
                if memory.assign(self) == -1:
                    raise RuntimeError(f"insufficient qubits in {node} to assign to {self}")

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
