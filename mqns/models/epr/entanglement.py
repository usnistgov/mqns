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

import hashlib
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar, Unpack, cast

import numpy as np

from mqns.models.core import QuantumModel
from mqns.models.qubit.const import OPERATOR_PAULI_I, QUBIT_STATE_0, QUBIT_STATE_P
from mqns.models.qubit.gate import CNOT, H, U, X, Y, Z
from mqns.models.qubit.qubit import QState, Qubit
from mqns.simulator import Time
from mqns.utils import get_rand

if TYPE_CHECKING:
    from mqns.entity.node import QNode


def _name_hash(s1: str) -> str:
    return hashlib.sha256(s1.encode()).hexdigest()


EntanglementT = TypeVar("EntanglementT", bound="BaseEntanglement")


class BaseEntanglementInitKwargs(TypedDict, total=False):
    name: str | None
    creation_time: Time
    decoherence_time: Time
    src: "QNode|None"
    dst: "QNode|None"
    mem_decohere_rate: tuple[float, float]


class BaseEntanglement(ABC, Generic[EntanglementT], QuantumModel):
    """Base entanglement model."""

    def __init__(self, **kwargs: Unpack[BaseEntanglementInitKwargs]):
        """
        Constructor.

        Args:
            name: Entanglement name, defaults to a random string.
            creation_time: EPR creation time point, defaults to `Time.SENTINEL`.
            decoherence_time: Qubits decoherence time point, defaults to `Time.SENTINEL`.
            src: Left node that holds one entangled qubit.
            dst: Right node that holds one entangled qubit.
            mem_decohere_rate: Memory decoherence rate at src and dst, defaults to (0.0,0.0).
        """
        name = kwargs.get("name")
        self.name = uuid.uuid4().hex if name is None else name
        """Descriptive name."""

        self.is_decoherenced = False
        """Whether the entanglement has decohered."""
        self.creation_time = kwargs.get("creation_time", Time.SENTINEL)
        """
        EPR creation time point assigned by LinkLayer or swapping.
        Some operations are unavailable if this is `Time.SENTINEL`.
        """
        self.decoherence_time = kwargs.get("decoherence_time", Time.SENTINEL)
        """
        EPR decoherence time point assigned by memory or swapping.
        Some operations are unavailable if this is `Time.SENTINEL`.
        """
        self.read = False
        """Whether the entanglement has been read from the memory by either node."""

        self.key: str | None = None
        """Reservation key used by LinkLayer."""
        self.src = kwargs.get("src")
        """One node that holds one entangled qubit, at the left side of a path."""
        self.dst = kwargs.get("dst")
        """The other node that holds the other entangled qubit, at the right side of a path."""
        self.mem_decohere_rate = kwargs.get("mem_decohere_rate", (0.0, 0.0))
        """Memory decoherence rate in Hz at src and dst."""

        self.ch_index = -1
        """
        Index of elementary entanglement in a path, smaller indices are on the left side.
        Negative means this is not an elementary entanglement.
        """
        self.orig_eprs: list[EntanglementT] = []
        """Elementary entanglements that swapped into this entanglement."""
        self.tmp_path_ids: frozenset[int] | None = None
        """Possible path IDs, used by MuxSchemeStatistical and MuxSchemeDynamicEpr."""

    @property
    @abstractmethod
    def fidelity(self) -> float:
        pass

    @fidelity.setter
    @abstractmethod
    def fidelity(self, value: float):
        pass

    @property
    def decoherence_rate(self) -> float:
        """Pair decoherence rate in Hz"""
        return sum(self.mem_decohere_rate)

    @classmethod
    def swap(
        cls,
        epr0: EntanglementT,
        epr1: EntanglementT,
        *,
        now: Time,
        ps=1.0,
    ) -> EntanglementT | None:
        """
        Perform swapping between `epr0` and `epr1`, and distribute a new entanglement.

        Args:
            epr0: left entanglement.
            epr1: right entanglement.
            now: current timestamp.
            ps: probability of successful swapping.

        Returns:
            New entanglement, or None if swap failed.
        """

        assert epr0.dst == epr1.src  # src and dst can be None

        if epr0.is_decoherenced or epr1.is_decoherenced:
            return None

        if ps < 1.0 and get_rand() >= ps:  # swap failed
            epr0.is_decoherenced = True
            epr1.is_decoherenced = True
            return None

        orig_eprs: list[EntanglementT] = []
        for epr in (epr0, epr1):
            if not epr.read:
                epr.read = True
                epr.store_error_model((now - epr.creation_time).sec, epr.decoherence_rate)

            if epr.ch_index > -1:
                orig_eprs.append(epr)
            else:
                orig_eprs.extend(cast(list[EntanglementT], epr.orig_eprs))

        ne = cls._make_swapped(
            epr0,
            epr1,
            name=_name_hash("-".join((e.name for e in orig_eprs))),
            creation_time=now,
            decoherence_time=min(epr0.decoherence_time, epr1.decoherence_time),
            src=epr0.src,
            dst=epr1.dst,
            mem_decohere_rate=(epr0.mem_decohere_rate[0], epr1.mem_decohere_rate[1]),
        )
        ne.orig_eprs = orig_eprs
        return ne

    @staticmethod
    @abstractmethod
    def _make_swapped(epr0: EntanglementT, epr1: EntanglementT, **kwargs: Unpack[BaseEntanglementInitKwargs]) -> EntanglementT:
        """
        Create a new entanglement that is the result of successful swapping between epr0 and epr1.
        Most properties are provided in kwargs or assigned subsequently.
        Subclass implementation should calculate the fidelity of new entanglement.
        """
        pass

    @abstractmethod
    def distillation(self, epr: EntanglementT) -> EntanglementT | None:
        """
        Use `self` and `epr` to perform distillation/purification and distribute a new entanglement.

        Args:
            epr: another entanglement.

        Returns:
            New entanglement.
        """
        pass

    def to_qubits(self) -> list[Qubit]:
        """
        Transport the entanglement into a pair of qubits based on the fidelity.
        Suppose the first qubit is [1/sqrt(2), 1/sqrt(2)].H

        Returns:
            A list of two qubits

        """
        if self.is_decoherenced:
            q0 = Qubit(state=QUBIT_STATE_P, name="q0")
            q1 = Qubit(state=QUBIT_STATE_P, name="q1")
            return [q0, q1]
        q0 = Qubit(state=QUBIT_STATE_0, name="q0")
        q1 = Qubit(state=QUBIT_STATE_0, name="q1")
        a = np.sqrt(self.fidelity / 2)
        b = np.sqrt((1 - self.fidelity) / 2)
        qs = QState([q0, q1], state=np.array([[a], [b], [b], [a]], dtype=np.complex128))
        q0.state = qs
        q1.state = qs
        self.is_decoherenced = True
        return [q0, q1]

    def teleportion(self, qubit: Qubit) -> Qubit:
        """
        Use `self` and `qubit` to perform teleportation.
        """
        q1, q2 = self.to_qubits()
        CNOT(qubit, q1)
        H(qubit)
        c0 = qubit.measure()
        c1 = q1.measure()
        if c1 == 1 and c0 == 0:
            X(q2)
        elif c1 == 0 and c0 == 1:
            Z(q2)
        elif c1 == 1 and c0 == 1:
            Y(q2)
            U(q2, np.complex128(1j) * OPERATOR_PAULI_I)
        self.is_decoherenced = True
        return q2

    def __repr__(self) -> str:
        return "<epr " + self.name + ">"
