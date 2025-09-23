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

from mqns.models.core.backend import QuantumModel
from mqns.models.epr.entanglement import BaseEntanglement


class BellStateEntanglement(BaseEntanglement["BellStateEntanglement"], QuantumModel):
    """`BellStateEntanglement` is the ideal max entangled qubits. Its fidelity is always 1."""

    def swapping(self, epr: "BellStateEntanglement", *, name: str | None = None, ps: float = 1) -> "BellStateEntanglement|None":
        ne = BellStateEntanglement(name=name)
        if self.is_decoherenced or epr.is_decoherenced:
            return None
        epr.is_decoherenced = True
        self.is_decoherenced = True
        return ne

    def distillation(self, epr: "BellStateEntanglement") -> "BellStateEntanglement":
        ne = BellStateEntanglement()
        if self.is_decoherenced or epr.is_decoherenced:
            ne.is_decoherenced = True
            ne.fidelity = 0
        epr.is_decoherenced = True
        self.is_decoherenced = True
        return ne

    def store_error_model(self, t: float = 0, decoherence_rate: float = 0, **kwargs):
        """The default error model for storing this entangled pair in a quantum memory.
        The default behavior is doing nothing

        Args:
            t: the time stored in a quantum memory. The unit it second.
            decoherence_rate (float): the decoherence_rate
            kwargs: other parameters

        """
        pass

    def transfer_error_model(self, length: float = 0, decoherence_rate: float = 0, **kwargs):
        """The default error model for transmitting this entanglement.
        The default behavior is doing nothing

        Args:
            length (float): the length of the channel
            decoherence_rate (float): the decoherency rate
            kwargs: other parameters

        """
        pass
