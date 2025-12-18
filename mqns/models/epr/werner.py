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


from typing import Unpack, overload, override

import numpy as np

from mqns.models.epr.entanglement import BaseEntanglement, BaseEntanglementInitKwargs
from mqns.models.qubit.const import QUBIT_STATE_0, QUBIT_STATE_P
from mqns.models.qubit.qubit import QState, Qubit
from mqns.utils import get_rand

phi_p: np.ndarray = 1 / np.sqrt(2) * np.array([[1], [0], [0], [1]])


def _fidelity_from_w(w: float) -> float:
    return (w * 3 + 1) / 4


def _fidelity_to_w(f: float) -> float:
    return (f * 4 - 1) / 3


_w_0 = _fidelity_to_w(0.0)
_w_1 = _fidelity_to_w(1.0)


class WernerStateEntanglement(BaseEntanglement["WernerStateEntanglement"]):
    """A pair of entangled qubits in Werner State with a hidden-variable."""

    @overload
    def __init__(self, *, fidelity: float = 1.0, **kwargs: Unpack[BaseEntanglementInitKwargs]):
        """Construct with fidelity."""
        pass

    @overload
    def __init__(self, *, w: float, **kwargs: Unpack[BaseEntanglementInitKwargs]):
        """Construct with Werner parameter."""
        pass

    def __init__(self, *, fidelity: float | None = None, w: float = _w_1, **kwargs: Unpack[BaseEntanglementInitKwargs]):
        super().__init__(**kwargs)
        self.w = _fidelity_to_w(fidelity) if fidelity is not None else w
        """Werner parameter."""
        assert _w_0 <= self.w <= _w_1

    @property
    def fidelity(self) -> float:
        return _fidelity_from_w(self.w)

    @fidelity.setter
    def fidelity(self, value: float):
        assert 0.0 <= value <= 1.0
        self.w = _fidelity_to_w(value)

    @staticmethod
    @override
    def _make_swapped(
        epr0: "WernerStateEntanglement", epr1: "WernerStateEntanglement", **kwargs: Unpack[BaseEntanglementInitKwargs]
    ):
        return WernerStateEntanglement(w=epr0.w * epr1.w, **kwargs)

    @override
    def distillation(self, epr: "WernerStateEntanglement") -> "WernerStateEntanglement|None":
        _ = epr
        raise NotImplementedError()

    def purify(self, epr: "WernerStateEntanglement") -> bool:
        """
        Use `self` and `epr` to perform distillation and update this entanglement.
        Using Bennett 96 protocol and estimate lower bound.

        Args:
            epr: another entanglement.

        Returns:
            Whether purification succeeded.
        """
        if self.is_decoherenced or epr.is_decoherenced:
            self.is_decoherenced = True
            self.w = _w_0
            return False

        epr.is_decoherenced = True
        fmin = min(self.fidelity, epr.fidelity)
        expr1 = fmin**2 + 5 / 9 * (1 - fmin) ** 2 + 2 / 3 * fmin * (1 - fmin)

        if get_rand() > expr1:
            self.is_decoherenced = True
            self.w = _w_0
            return False

        self.fidelity = (fmin**2 + (1 - fmin) ** 2 / 9) / expr1
        return True

    @override
    def store_error_model(self, t: float = 0, decoherence_rate: float = 0, **kwargs):
        """
        Apply an error model for storing this entangled pair in quantum memory::

            w = w * e^{-decoherence_rate * t}

        Args:
            t: the time stored in a quantum memory in seconds.
            decoherence_rate: the decoherence rate, equals to the inverse of coherence time.

        """
        _ = kwargs
        self.w *= np.exp(-decoherence_rate * t)

    @override
    def transfer_error_model(self, length: float = 0, decoherence_rate: float = 0, **kwargs):
        """
        Apply an error model for transmitting this entanglement::

            w = w * e^{decoherence_rate * length}

        Args:
            length: the length of the channel in kilometers.
            decoherence_rate: the decoherence rate, equals to the inverse of coherence time.

        """
        _ = kwargs
        self.w *= np.exp(-decoherence_rate * length)

    @override
    def to_qubits(self) -> list[Qubit]:
        if self.is_decoherenced:
            q0 = Qubit(state=QUBIT_STATE_P, name="q0")
            q1 = Qubit(state=QUBIT_STATE_P, name="q1")
            return [q0, q1]

        q0 = Qubit(state=QUBIT_STATE_0, name="q0")
        q1 = Qubit(state=QUBIT_STATE_0, name="q1")

        rho = self.w * np.dot(phi_p, phi_p.T.conjugate()) + (1 - self.w) / 4 * np.identity(4)
        qs = QState([q0, q1], rho=rho)
        q0.state = qs
        q1.state = qs
        self.is_decoherenced = True
        return [q0, q1]

    @override
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, fidelity={self.fidelity:.4f}, "
            f"is_decoherenced={self.is_decoherenced}, "
            f"src={self.src}, dst={self.dst}, "
            f"ch_index={self.ch_index}, "
            f"orig_eprs={[e.name for e in self.orig_eprs]}), "
            f"creation_time={self.creation_time}, "
            f"decoherence_time={self.decoherence_time}), "
            f"tmp_path_ids={self.tmp_path_ids})"
        )
