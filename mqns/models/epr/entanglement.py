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
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import numpy as np

from mqns.models.qubit.const import OPERATOR_PAULI_I, QUBIT_STATE_0, QUBIT_STATE_P
from mqns.models.qubit.gate import CNOT, H, U, X, Y, Z
from mqns.models.qubit.qubit import QState, Qubit
from mqns.simulator import Time

if TYPE_CHECKING:
    from mqns.entity.node import QNode


def _name_hash(s1: str) -> str:
    return hashlib.sha256(s1.encode()).hexdigest()


EntanglementT = TypeVar("EntanglementT")


class BaseEntanglement(ABC, Generic[EntanglementT]):
    """Base entanglement model."""

    def __init__(self, *, name: str | None = None):
        """
        Constructor.

        Args:
            name: the entanglement name, defaults to a random string.
        """
        self.name = uuid.uuid4().hex if name is None else name
        """Descriptive name."""
        self.key: str | None = None
        """Reservation key used by LinkLayer."""
        self.is_decoherenced = False
        """Whether the entanglement has decohered."""
        self.creation_time: Time | None = None
        """Entanglement creation time assigned by LinkLayer or swapping."""
        self.decoherence_time: Time | None = None
        """Entanglement decoherence time assigned by memory or swapping."""
        self.read = False
        """Whether the entanglement has been read from the memory by either node."""
        self.src: "QNode|None" = None
        """One node that holds one entangled qubit, at the left side of a path."""
        self.dst: "QNode|None" = None
        """The other node that holds the other entangled qubit, at the right side of a path."""
        self.ch_index = -1
        """Index of this entanglement in a path, smaller indices are on the left side."""
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

    @abstractmethod
    def swapping(self, epr: EntanglementT, *, name: str | None = None, ps: float = 1) -> EntanglementT | None:
        """
        Use `self` and `epr` to perform swapping and distribute a new entanglement.

        Args:
            epr: another entanglement.
            name: name of the new entanglement.
            ps: probability of successful swapping.

        Returns:
            New entanglement.
        """
        pass

    def _update_orig_eprs(self, *eprs: "BaseEntanglement", update_name=False) -> None:
        merged: dict[str, BaseEntanglement] = {}
        for epr in eprs:
            if epr.ch_index > -1:  # this is an elementary epr
                merged[epr.name] = epr
            else:  # this is not an elementary epr
                for oe in cast(list[BaseEntanglement], epr.orig_eprs):
                    merged[oe.name] = oe

        orig_eprs = sorted(merged.values(), key=lambda e: e.ch_index)
        self.orig_eprs = cast(list[EntanglementT], orig_eprs)

        if update_name:
            epr_names = (e.name for e in orig_eprs)
            self.name = _name_hash("-".join(epr_names))

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
        if self.name is not None:
            return "<epr " + self.name + ">"
        return super().__repr__()
