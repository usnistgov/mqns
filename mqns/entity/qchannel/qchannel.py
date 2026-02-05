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

import copy
from typing import Any, Unpack, final, override

from mqns.entity.base_channel import BaseChannel, BaseChannelInitKwargs, calc_transmission_prob
from mqns.entity.node import QNode
from mqns.entity.qchannel.link_arch import LinkArch
from mqns.entity.qchannel.link_arch_dim import LinkArchDimBkSeq
from mqns.models.core import QuantumModel
from mqns.models.epr import Entanglement
from mqns.models.error import DepolarErrorModel
from mqns.models.error.input import ErrorModelInputLength, parse_error
from mqns.simulator import Event, Time


class QuantumChannelInitKwargs(BaseChannelInitKwargs, total=False):
    link_arch: LinkArch
    """Link architecture model."""
    alpha: float
    """
    Fiber attenuation loss in dB/km.

    If ``BaseChannel.drop_rate`` is zero but both ``length`` and ``alpha`` are positive,
    ``BaseChannel.drop_rate`` is recalculated from ``length`` and ``alpha``.
    """
    transfer_error: ErrorModelInputLength
    """Transfer error model for loss of quantum information."""


class QuantumChannel(BaseChannel[QNode]):
    """
    QuantumChannel is the channel for transmitting photonic qubits.

    In entanglement routing experiments, MQNS does not use the ``send()`` method to transmit photonic qubits.
    Instead, the ``LinkLayer`` application calculates entanglement arrival times and fidelity from channel parameters
    such as ``length`` and ``link_arch``, and directly schedules entanglement arrivals.
    """

    def __init__(self, name: str, **kwargs: Unpack[QuantumChannelInitKwargs]):
        super().__init__(name, **kwargs)

        link_arch = kwargs.get("link_arch", None)
        self.link_arch = copy.deepcopy(link_arch) if link_arch else LinkArchDimBkSeq()
        """Link architecture model (separate instance per channel)."""

        self.alpha = kwargs.get("alpha", 0.0)
        """Fiber attenuation loss in dB/km."""
        assert self.alpha >= 0
        if self.drop_rate == 0 and self.length > 0 and self.alpha > 0:
            self.drop_rate = 1 - calc_transmission_prob(self.length, self.alpha)

        self.transfer_error = parse_error(kwargs.get("transfer_error"), DepolarErrorModel, self.length)
        """
        Transfer error model.

        It reflects loss of quantum information when a qubit/EPR is sent through the fiber.
        It does not reflect loss of photons.
        """

    @override
    def handle(self, event: Event):
        raise RuntimeError(f"unexpected event {event}")

    def assign_memory_qubits(self, *, capacity: int | dict[str, int] = 1):
        """
        Assign memory qubits at each node connected to the qchannel.

        Args:
            capacity: required quantity of qubits.
                      If given as an integer, this applies to every node.
                      If given as a dict, it should be a mapping from node name to capacity of this node,
                      where every node connected to the qchannel must appear in the dict.

        Raises:
            OverflowError - insufficient qubits.
        """
        for node in self.node_list:
            cap = capacity if isinstance(capacity, int) else capacity[node.name]
            node.memory.assign(self, n=cap)

    def send(self, qubit: QuantumModel, next_hop: QNode):
        """
        Send a qubit to the next_hop.

        Args:
            qubit: the photonic qubit.
            next_hop: the recipient quantum node.

        Raises:
            NextHopNotConnectionException: next_hop is not connected to this channel.
        """
        drop, recv_time = self._send(
            packet_repr=f"qubit {qubit}",
            packet_len=1,
            next_hop=next_hop,
        )

        if drop:
            # photon is lost -> flag this pair as decoherenced to inform receiver node
            if isinstance(qubit, Entanglement):
                qubit.is_decoherenced = True
            return

        # operation on the qubit
        qubit.apply_error(self.transfer_error)
        send_event = RecvQubitPacket(t=recv_time, by=self, qchannel=self, qubit=qubit, dest=next_hop)
        self.simulator.add_event(send_event)

    def __repr__(self) -> str:
        return "<qchannel " + self.name + ">"


@final
class RecvQubitPacket(Event):
    """
    Event dispatched on recipient QNode for receiving a qubit.
    """

    def __init__(
        self, *, t: Time, name: str | None = None, by: Any = None, qchannel: QuantumChannel, qubit: QuantumModel, dest: QNode
    ):
        super().__init__(t=t, name=name, by=by)
        self.qchannel = qchannel
        self.qubit = qubit
        self.dest = dest

    @override
    def invoke(self) -> None:
        self.dest.handle(self)
