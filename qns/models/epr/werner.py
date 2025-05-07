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

from typing import Optional, List
from qns.models.epr.entanglement import BaseEntanglement
from qns.models.core.backend import QuantumModel
from qns.models.qubit.qubit import Qubit, QState
from qns.models.qubit.const import QUBIT_STATE_0, QUBIT_STATE_P
import numpy as np

from qns.utils.rnd import get_rand

import hashlib

def hash(s1: str) -> str:
    return hashlib.sha256(s1.encode()).hexdigest()


class WernerStateEntanglement(BaseEntanglement, QuantumModel):
    """
    `WernerStateEntanglement` is a pair of entangled qubits in Werner State with a hidden-variable.
    """
    def __init__(self, fidelity: float = 1, name: Optional[str] = None):
        """
        generate an entanglement with certain fidelity

        Args:
            fidelity (float): the fidelity
            name (str): the entanglement name
        """
        self.w = (fidelity * 4 - 1) / 3
        self.name = name
        self.is_decoherenced = False
        self.src = None
        self.dst = None
        self.ch_index = -1
        self.orig_eprs = []
        
        self.decoherence_time = None
        self.rcvd = 0
        self.key = None

    def __deepcopy__(self, memo):
        """ new_obj = WernerStateEntanglement(self.fidelity, self.name)
        new_obj.is_decoherenced = self.is_decoherenced
        new_obj.src = self.src
        new_obj.dst = self.dst
        new_obj.ch_index = self.ch_index
        new_obj.orig_eprs = self.orig_eprs
        new_obj.decoherence_time = self.decoherence_time
        new_obj.rcvd = self.rcvd """
        return self

    @property
    def fidelity(self) -> float:
        return (self.w * 3 + 1) / 4

    @fidelity.setter
    def fidelity(self, fidelity: float = 1):
        self.w = (fidelity * 4 - 1) / 3

    def swapping(self, epr: "WernerStateEntanglement", name: Optional[str] = None, ps: float = 1) -> "WernerStateEntanglement":
        """
        Use `self` and `epr` to perfrom swapping and distribute a new entanglement

        Args:
            epr (WernerEntanglement): another entanglement
            name (str): the name of the new entanglement
        Returns:
            the new distributed entanglement
        """
        ne = WernerStateEntanglement()
        if self.is_decoherenced or epr.is_decoherenced:
            return None

        r = get_rand()
        if r >= ps:              # swap failed
            epr.is_decoherenced = True
            self.is_decoherenced = True
            return None

        ne.w = self.w * epr.w
        ne.orig_eprs = self.merge_orig_eprs(epr)
        
        eprs_name_list = [e.name for e in ne.orig_eprs]
        ne.name = hash('-'.join(eprs_name_list))
        
        # set decoherence time to the shorter among the two pairs
        ne.decoherence_time = min(self.decoherence_time, epr.decoherence_time)
        return ne

    def distillation(self, epr: "WernerStateEntanglement", name: Optional[str] = None):
        """
        Use `self` and `epr` to perfrom distillation and distribute a new entanglement.
        Using Bennett 96 protocol and estimate lower bound.

        Args:
            epr (WernerEntanglement): another entanglement
            name (str): the name of the new entanglement
        Returns:
            the new distributed entanglement
        """
        ne = WernerStateEntanglement()
        if self.is_decoherenced or epr.is_decoherenced:
            ne.is_decoherenced = True
            ne.fidelity = 0
            return
        epr.is_decoherenced = True
        self.is_decoherenced = True
        fmin = min(self.fidelity, epr.fidelity)

        if get_rand() > (fmin ** 2 + 5 / 9 * (1 - fmin) ** 2 + 2 / 3 * fmin * (1 - fmin)):
            ne.is_decoherenced = True
            ne.fidelity = 0
            return
        ne.fidelity = (fmin ** 2 + (1 - fmin) ** 2 / 9) /\
                      (fmin ** 2 + 5 / 9 * (1 - fmin) ** 2 + 2 / 3 * fmin * (1 - fmin))
        return ne

    def store_error_model(self, t: float, decoherence_rate: Optional[float], **kwargs):
        """
        The default error model for storing this entangled pair in a quantum memory.
        The default behavior is: w = w*e^{-decoherence_rate*t}, default a = 0

        Args:
            t: the time stored in a quantum memory. The unit it second.
            decoherence_rate: the decoherence rate, equals to 1/T_coh, where T_coh is the coherence time.
            kwargs: other parameters
        """
        self.w = self.w * np.exp(-decoherence_rate * t)

    def transfer_error_model(self, length: float, decoherence_rate: Optional[float] = 0, **kwargs):
        """
        The default error model for transmitting this entanglement.
        The success possibility of transmitting is: w = w* e^{decoherence_rate * length}

        Args:
            length (float): the length of the channel
            kwargs: other parameters
        """
        self.w = self.w * np.exp(-decoherence_rate * length)

    def to_qubits(self) -> List[Qubit]:
        if self.is_decoherenced:
            q0 = Qubit(state=QUBIT_STATE_P, name="q0")
            q1 = Qubit(state=QUBIT_STATE_P, name="q1")
            return [q0, q1]

        q0 = Qubit(state=QUBIT_STATE_0, name="q0")
        q1 = Qubit(state=QUBIT_STATE_0, name="q1")

        phi_p = 1/np.sqrt(2) * np.array([[1], [0], [0], [1]])
        rho = self.w * np.dot(phi_p, phi_p.T.conjugate()) + (1-self.w)/4 * np.identity(4)
        print(rho)
        qs = QState([q0, q1], rho=rho)
        q0.state = qs
        q1.state = qs
        self.is_decoherenced = True
        return [q0, q1]
    
    
    def merge_orig_eprs(self, epr):
        # Helper: get a dict of name -> epr from an object's orig_epr list
        def epr_dict(obj):
            return { e.name: e for e in obj.orig_eprs }

        # Merge by name
        merged = epr_dict(self)
        for name, epr in epr_dict(epr).items():
            if name not in merged:
                merged[name] = epr

        # Add elementary eprs     
        if self.ch_index > -1 and self.name not in merged:
            merged['self'] = self
        if epr.ch_index > -1 and epr.name not in merged:
            merged['epr'] = epr

        # Sort the result by epr.index
        return sorted(
            merged.values(),
            key=lambda e: e.ch_index
        )