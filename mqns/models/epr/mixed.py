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

from collections.abc import Iterable
from typing import Unpack, final, overload, override

import numpy as np

from mqns.models.epr.entanglement import Entanglement, EntanglementInitKwargs
from mqns.models.qubit.state import BELL_RHO_PHI_N, BELL_RHO_PHI_P, BELL_RHO_PSI_N, BELL_RHO_PSI_P, QubitRho, check_qubit_rho
from mqns.utils import get_rand


@final
class MixedStateEntanglement(Entanglement["MixedStateEntanglement"]):
    """
    `MixedStateEntanglement` is a pair of entangled qubits in mixed State with a hidden-variable.

    ::

        rho = A * Phi^+ + B * Psi^+ + C * Psi^- + D * Phi^-
    """

    @overload
    def __init__(self, *, fidelity=1.0, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Construct with fidelity, used as the probability of ``Phi^+``.
        """
        pass

    @overload
    def __init__(self, *, a: float, b: float, c: float, d: float, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Construct with four probability values.

        Args:
            a: Probability of ``Phi^+``.
            b: Probability of ``Psi^+``.
            c: Probability of ``Psi^-``.
            d: Probability of ``Phi^-``.
        """
        pass

    def __init__(self, *, fidelity: float | None = None, a=1.0, b=0.0, c=0.0, d=0.0, **kwargs: Unpack[EntanglementInitKwargs]):
        super().__init__(**kwargs)
        if fidelity is None:
            self._set_probabilities(a, b, c, d)
        else:
            bcd = (1 - fidelity) / 3
            self._set_probabilities(fidelity, bcd, bcd, bcd)

    @property
    @override
    def fidelity(self) -> float:
        return self.a

    @fidelity.setter
    @override
    def fidelity(self, value: float):
        assert 0 <= value <= 1
        self.a = value

    def _set_probabilities(self, a: float, b: float, c: float, d: float) -> None:
        total = a + b + c + d
        self.a = a / total
        """Probability of ``Phi^+``."""
        self.b = b / total
        """Probability of ``Phi^-``."""
        self.c = c / total
        """Probability of ``Psi^+``."""
        self.d = d / total
        """Probability of ``Psi^-``."""

    @staticmethod
    @override
    def _make_swapped(epr0: "MixedStateEntanglement", epr1: "MixedStateEntanglement", **kwargs: Unpack[EntanglementInitKwargs]):
        return MixedStateEntanglement(
            a=epr0.a * epr1.a + epr0.b * epr1.b + epr0.c * epr1.c + epr0.d * epr1.d,
            b=epr0.a * epr1.b + epr0.b * epr1.a + epr0.c * epr1.d + epr0.d * epr1.c,
            c=epr0.a * epr1.c + epr0.b * epr1.d + epr0.c * epr1.a + epr0.d * epr1.b,
            d=epr0.a * epr1.d + epr0.b * epr1.c + epr0.c * epr1.d + epr0.d * epr1.a,
            **kwargs,
        )

    @override
    def _do_purify(self, epr1: "MixedStateEntanglement") -> bool:
        """
        Perform distillation using BBPSSW protocol.
        """
        p_succ = (self.a + self.d) * (epr1.a + epr1.d) + (self.b + self.c) * (epr1.c + epr1.b)
        if get_rand() > p_succ:
            return False

        self._set_probabilities(
            (self.a * epr1.a + self.d * epr1.d) / p_succ,
            (self.b * epr1.b + self.c * epr1.c) / p_succ,
            (self.b * epr1.c + self.c * epr1.b) / p_succ,
            (self.a * epr1.d + self.d * epr1.a) / p_succ,
        )
        return True

    @override
    def store_error_model(self, t: float = 0, decoherence_rate: float = 0, **kwargs):
        """
        The default error model for storing this entangled pair in a quantum memory.
        The default behavior is::

            a = 0.25 + (a-0.25)*e^{decoherence_rate*t}
            b = 0.25 + (b-0.25)*e^{decoherence_rate*t}
            c = 0.25 + (c-0.25)*e^{decoherence_rate*t}
            d = 0.25 + (d-0.25)*e^{decoherence_rate*t}

        Args:
            t: the time stored in a quantum memory. The unit it second.
            decoherence_rate: the decoherence rate, equals to 1/T_coh, where T_coh is the coherence time.
            kwargs: other parameters

        """
        self._set_probabilities(
            0.25 + (self.a - 0.25) * np.exp(-decoherence_rate * t),
            0.25 + (self.b - 0.25) * np.exp(-decoherence_rate * t),
            0.25 + (self.c - 0.25) * np.exp(-decoherence_rate * t),
            0.25 + (self.d - 0.25) * np.exp(-decoherence_rate * t),
        )

    @override
    def transfer_error_model(self, length: float = 0, decoherence_rate: float = 0, **kwargs):
        """
        The default error model for transmitting this entanglement.
        The success possibility of transmitting is::

            a = 0.25 + (a-0.25)*e^{decoherence_rate*length}
            b = 0.25 + (b-0.25)*e^{decoherence_rate*length}
            c = 0.25 + (c-0.25)*e^{decoherence_rate*length}
            d = 0.25 + (d-0.25)*e^{decoherence_rate*length}

        Args:
            length (float): the length of the channel
            decoherence_rate (float): the decoherency rate
            kwargs: other parameters

        """
        self._set_probabilities(
            0.25 + (self.a - 0.25) * np.exp(-decoherence_rate * length),
            0.25 + (self.b - 0.25) * np.exp(-decoherence_rate * length),
            0.25 + (self.c - 0.25) * np.exp(-decoherence_rate * length),
            0.25 + (self.d - 0.25) * np.exp(-decoherence_rate * length),
        )

    @override
    def _to_qubits_rho(self) -> QubitRho:
        return check_qubit_rho(
            self.a * BELL_RHO_PHI_P + self.b * BELL_RHO_PSI_P + self.c * BELL_RHO_PSI_N + self.d * BELL_RHO_PHI_N, n=2
        )

    @override
    def _describe_fidelity(self) -> Iterable[str]:
        yield f"a={self.a:.4f}"
        yield f"b={self.b:.4f}"
        yield f"c={self.c:.4f}"
        yield f"d={self.d:.4f}"
