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


from typing import Unpack, final, overload, override

import numpy as np

from mqns.models.epr.const import RHO_PHI_P
from mqns.models.epr.entanglement import Entanglement, EntanglementInitKwargs
from mqns.models.qubit.typing import MultiQubitRho
from mqns.utils import get_rand


def _fidelity_from_w(w: float) -> float:
    return (w * 3 + 1) / 4


def _fidelity_to_w(f: float) -> float:
    return (f * 4 - 1) / 3


_w_0 = _fidelity_to_w(0.0)
_w_1 = _fidelity_to_w(1.0)


@final
class WernerStateEntanglement(Entanglement["WernerStateEntanglement"]):
    """A pair of entangled qubits in Werner State with a hidden-variable."""

    @overload
    def __init__(self, *, fidelity: float = 1.0, **kwargs: Unpack[EntanglementInitKwargs]):
        """Construct with fidelity."""
        pass

    @overload
    def __init__(self, *, w: float, **kwargs: Unpack[EntanglementInitKwargs]):
        """Construct with Werner parameter."""
        pass

    def __init__(self, *, fidelity: float | None = None, w: float = _w_1, **kwargs: Unpack[EntanglementInitKwargs]):
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

    @override
    def _mark_decoherenced(self) -> None:
        self.is_decoherenced = True
        self.w = _w_0

    @staticmethod
    @override
    def _make_swapped(
        epr0: "WernerStateEntanglement", epr1: "WernerStateEntanglement", **kwargs: Unpack[EntanglementInitKwargs]
    ):
        return WernerStateEntanglement(w=epr0.w * epr1.w, **kwargs)

    @override
    def _do_purify(self, epr1: "WernerStateEntanglement") -> bool:
        """
        Perform distillation using Bennett 96 protocol and estimate lower bound.
        """
        fmin = min(self.fidelity, epr1.fidelity)
        expr1 = fmin**2 + 5 / 9 * (1 - fmin) ** 2 + 2 / 3 * fmin * (1 - fmin)

        if get_rand() > expr1:
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
    def _to_qubits_rho(self) -> MultiQubitRho:
        return self.w * RHO_PHI_P + (1 - self.w) / 4 * np.identity(4)

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
