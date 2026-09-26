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

from collections.abc import Sequence
from typing import Final, Unpack, final, override

from mqns.entity.base_channel import BaseChannel, BaseChannelInitKwargs, calc_transmission_prob
from mqns.entity.node import QNode
from mqns.entity.qchannel.link_arch_input import LinkArchInput, parse_link_arch
from mqns.models.core import QuantumModel
from mqns.models.epr import Entanglement
from mqns.models.error import DepolarErrorModel
from mqns.models.error.input import ErrorModelInputBasic, ErrorModelInputLength, parse_error
from mqns.simulator import Event, Time


class QuantumChannelInitKwargs(BaseChannelInitKwargs, total=False):
    link_arch: LinkArchInput
    """Link architecture model."""
    alpha: float
    """
    Fiber attenuation loss in dB/km.

    If ``BaseChannel.drop_rate`` is zero but both ``length`` and ``alpha`` are positive,
    ``BaseChannel.drop_rate`` is recalculated from ``length`` and ``alpha``.

    In ``LinkArch``, this parameter determines the success probability,
    but does not affect the decoherence / quality of the state given the photon arrived.
    """
    eta_s: float
    """Source efficiency between 0 and 1, defaults to ``0.95``."""
    eta_d: float
    """Detector efficiency between 0 and 1, defaults to ``0.95``."""
    init_fidelity: float | Sequence[float] | None
    """
    Initial fidelity value for entanglements delivered by this channel.

    If set as float, every entanglement has a Werner state with the specified fidelity.
    If set as four floats, every entanglement has a Bell-diagonal state with the specified I,Z,X,Y values.
    In these cases, the specified fidelity must be tuned to the EPR creation time,
    which is generally when the first photon is emitted or absorbed by a memory.

    If omitted or ``None``, ``LinkArch`` performs a mini simulation to determine the proper fidelity values,
    by applying error models to the memories, fibers, etc.
    """
    transfer_error: ErrorModelInputLength
    """
    Transfer error model for loss of quantum information.

    In ``LinkArch``, this parameter determines the decoherence / quality of the state
    given the photon arrived, but does not affect the success probability.
    """
    bsa_error: ErrorModelInputBasic
    """
    Bell-state analyzer or absorptive memory capture error model, defaults to perfect.

    In ``LinkArch``, this parameter determines the decoherence / quality of the state
    given the photon arrived, but does not affect the success probability.
    """


class QuantumChannel(BaseChannel[QNode]):
    """
    A channel for transmitting photonic qubits.

    In entanglement routing experiments, MQNS does not use the ``send()`` method to transmit photonic qubits.
    Instead, the ``LinkLayer`` application calculates entanglement arrival times and fidelity from channel parameters
    such as ``length`` and ``link_arch``, and directly schedules entanglement arrivals.

    This class implements the protocol ``mqns.entity.qchannel.ChannelParameters``.
    """

    def __init__(self, name: str, **kwargs: Unpack[QuantumChannelInitKwargs]):
        super().__init__(name, **kwargs)

        self.link_arch: Final = parse_link_arch(kwargs.get("link_arch", None))
        """Link architecture model."""

        self.alpha = kwargs.get("alpha", 0.0)
        assert self.alpha >= 0
        if self.drop_rate == 0 and self.length > 0 and self.alpha > 0:
            self.drop_rate = 1 - calc_transmission_prob(self.length, self.alpha)
        self.eta_s = kwargs.get("eta_s", 0.95)
        self.eta_d = kwargs.get("eta_d", 0.95)

        self.init_fidelity = kwargs.get("init_fidelity")
        self.transfer_error = parse_error(kwargs.get("transfer_error"), DepolarErrorModel, self.length)
        self.bsa_error = parse_error(kwargs.get("bsa_error"), DepolarErrorModel, -1)

    def assign_memory_qubits(self, *, capacity: int | dict[str, int] = 1, path_id: int | None = None):
        """
        Assign memory qubits at each node connected to the qchannel.

        Args:
            capacity: Required quantity of qubits.
                      If given as an integer, this applies to every node.
                      If given as a dict, it should be a mapping from node name to capacity of this node,
                      where every node connected to the qchannel must appear in the dict.
            path_id: If not ``None``, also allocate the qubits to the specified path.

        Raises:
            OverflowError: insufficient qubits.
        """
        from mqns.entity.memory import PathDirection  # noqa: PLC0415

        for node, dir in zip(self.node_list, (PathDirection.R, PathDirection.L), strict=True):
            cap = capacity if isinstance(capacity, int) else capacity[node.name]
            node.memory.assign(self, n=cap)
            if path_id is not None:
                node.memory.allocate(self, path_id, dir, n=cap)

    def send(self, qubit: QuantumModel, next_hop: QNode):
        """
        Send a qubit to the next_hop.

        Args:
            qubit: The photonic qubit.
            next_hop: The recipient quantum node.

        Raises:
            LookupError: The next_hop is not connected to this channel.
        """
        drop, recv_time = self._send(qubit, 1, next_hop)

        if drop:
            # photon is lost -> flag this pair as decohered to inform receiver node
            if isinstance(qubit, Entanglement):
                qubit.is_decohered = True
            return

        # operation on the qubit
        qubit.apply_error(self.transfer_error)

        event = RecvQubitPacket(self, qubit, next_hop, t=recv_time)
        self.simulator.sched(event)

    def __repr__(self) -> str:
        return "<qchannel " + self.name + ">"


@final
class RecvQubitPacket(Event):
    """
    Event dispatched on recipient QNode for receiving a qubit.
    """

    def __init__(self, qchannel: QuantumChannel, qubit: QuantumModel, dest: QNode, *, t: Time):
        super().__init__(t)
        self.qchannel = qchannel
        self.qubit = qubit
        self.dest = dest

    @override
    def invoke(self) -> None:
        self.dest.handle(self)
