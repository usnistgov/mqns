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

import numpy as np

from mqns.models.core.backend import QuantumModel
from mqns.models.epr.entanglement import BaseEntanglement
from mqns.models.qubit.const import QUBIT_STATE_0, QUBIT_STATE_P
from mqns.models.qubit.qubit import QState, Qubit
from mqns.utils.rnd import get_rand


class MixedStateEntanglement(BaseEntanglement["MixedStateEntanglement"], QuantumModel):
    """`MixedStateEntanglement` is a pair of entangled qubits in mixed State with a hidden-variable.
    rho = A * Phi^+ + B * Psi^+ + C * Psi^- + D * Phi^-
    """

    def __init__(
        self,
        fidelity: float = 1,
        b: float | None = None,
        c: float | None = None,
        d: float | None = None,
        name: str | None = None,
    ):
        """Generate an entanglement with certain fidelity

        Args:
            fidelity (float): the fidelity, equals to the probability of Phi^+
            b (float): probability of Psi^+
            c (float): probability of Psi^-
            d (float): probability of Phi^-
            name (str): the entanglement name

        """
        super().__init__(fidelity=fidelity, name=name)
        self.b = b if b is not None else (1 - fidelity) / 3
        self.c = c if c is not None else (1 - fidelity) / 3
        self.d = d if d is not None else (1 - fidelity) / 3
        self.normalized()

    @property
    def a(self) -> float:
        """A equals to the fidelity"""
        return self.fidelity

    @a.setter
    def a(self, fidelity: float = 1):
        self.fidelity = fidelity

    def normalized(self):
        total = self.a + self.b + self.c + self.d
        # Normalized: a + b + c + d = 1
        self.a = self.a / total
        self.b = self.b / total
        self.c = self.c / total
        self.d = self.d / total

    def swapping(
        self, epr: "MixedStateEntanglement", *, name: str | None = None, ps: float = 1
    ) -> "MixedStateEntanglement|None":
        """Use `self` and `epr` to perform swapping and distribute a new entanglement

        Args:
            epr (MixedEntanglement): another entanglement
            name (str): the name of the new entanglement
        Returns:
            the new distributed entanglement

        """
        ne = MixedStateEntanglement(name=name)
        if self.is_decoherenced or epr.is_decoherenced:
            return None
        epr.is_decoherenced = True
        self.is_decoherenced = True

        r = get_rand()
        if r > ps:  # swap failed
            return None

        ne.a = self.a * epr.a + self.b * epr.b + self.c * epr.c + self.d * epr.d
        ne.b = self.a * epr.b + self.b * epr.a + self.c * epr.d + self.d * epr.c
        ne.c = self.a * epr.c + self.b * epr.d + self.c * epr.a + self.d * epr.b
        ne.d = self.a * epr.d + self.b * epr.c + self.c * epr.d + self.d * epr.a
        ne.normalized()
        return ne

    def distillation(self, epr: "MixedStateEntanglement", name: str | None = None):
        """Use `self` and `epr` to perform distillation and distribute a new entanglement.
        Using BBPSSW protocol.

        Args:
            epr (BaseEntanglement): another entanglement
            name (str): the name of the new entanglement
        Returns:
            the new distributed entanglement

        """
        ne = MixedStateEntanglement()
        if self.is_decoherenced or epr.is_decoherenced:
            ne.is_decoherenced = True
            ne.fidelity = 0
            return
        epr.is_decoherenced = True
        self.is_decoherenced = True
        p_succ = (self.a + self.d) * (epr.a + epr.d) + (self.b + self.c) * (epr.c + epr.b)

        if get_rand() > p_succ:
            ne.is_decoherenced = True
            ne.fidelity = 0
            return
        ne.a = (self.a * epr.a + self.d * epr.d) / p_succ
        ne.b = (self.b * epr.b + self.c * epr.c) / p_succ
        ne.c = (self.b * epr.c + self.c * epr.b) / p_succ
        ne.d = (self.a * epr.d + self.d * epr.a) / p_succ
        ne.normalized()
        return ne

    def store_error_model(self, t: float = 0, decoherence_rate: float = 0, **kwargs):
        """The default error model for storing this entangled pair in a quantum memory.
        The default behavior is:
            a = 0.25 + (a-0.25)*e^{decoherence_rate*t}
            b = 0.25 + (b-0.25)*e^{decoherence_rate*t}
            c = 0.25 + (c-0.25)*e^{decoherence_rate*t}
            d = 0.25 + (d-0.25)*e^{decoherence_rate*t}

        Args:
            t: the time stored in a quantum memory. The unit it second.
            decoherence_rate: the decoherence rate, equals to 1/T_coh, where T_coh is the coherence time.
            kwargs: other parameters

        """
        self.a = 0.25 + (self.a - 0.25) * np.exp(-decoherence_rate * t)
        self.b = 0.25 + (self.b - 0.25) * np.exp(-decoherence_rate * t)
        self.c = 0.25 + (self.c - 0.25) * np.exp(-decoherence_rate * t)
        self.d = 0.25 + (self.d - 0.25) * np.exp(-decoherence_rate * t)
        self.normalized()

    def transfer_error_model(self, length: float = 0, decoherence_rate: float = 0, **kwargs):
        """The default error model for transmitting this entanglement.
        The success possibility of transmitting is:
            a = 0.25 + (a-0.25)*e^{decoherence_rate*length}
            b = 0.25 + (b-0.25)*e^{decoherence_rate*length}
            c = 0.25 + (c-0.25)*e^{decoherence_rate*length}
            d = 0.25 + (d-0.25)*e^{decoherence_rate*length}

        Args:
            length (float): the length of the channel
            decoherence_rate (float): the decoherency rate
            kwargs: other parameters

        """
        self.a = 0.25 + (self.a - 0.25) * np.exp(-decoherence_rate * length)
        self.b = 0.25 + (self.b - 0.25) * np.exp(-decoherence_rate * length)
        self.c = 0.25 + (self.c - 0.25) * np.exp(-decoherence_rate * length)
        self.d = 0.25 + (self.d - 0.25) * np.exp(-decoherence_rate * length)
        self.normalized()

    def to_qubits(self) -> list[Qubit]:
        if self.is_decoherenced:
            q0 = Qubit(state=QUBIT_STATE_P, name="q0")
            q1 = Qubit(state=QUBIT_STATE_P, name="q1")
            return [q0, q1]

        q0 = Qubit(state=QUBIT_STATE_0, name="q0")
        q1 = Qubit(state=QUBIT_STATE_0, name="q1")

        phi_p = 1 / np.sqrt(2) * np.array([[1], [0], [0], [1]])
        phi_n = 1 / np.sqrt(2) * np.array([[1], [0], [0], [-1]])
        psi_p = 1 / np.sqrt(2) * np.array([[0], [1], [1], [0]])
        psi_n = 1 / np.sqrt(2) * np.array([[0], [1], [-1], [0]])
        rho = (
            self.a * np.dot(phi_p, phi_p.T.conjugate())
            + self.b * np.dot(psi_p, psi_p.T.conjugate())
            + self.c * np.dot(psi_n, psi_n.T.conjugate())
            + self.d * np.dot(phi_n, phi_n.T.conjugate())
        )

        qs = QState([q0, q1], rho=rho)
        q0.state = qs
        q1.state = qs
        self.is_decoherenced = True
        return [q0, q1]
