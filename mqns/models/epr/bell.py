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

from typing import final

from typing_extensions import override

from mqns.models.epr.entanglement import BaseEntanglement


@final
class BellStateEntanglement(BaseEntanglement["BellStateEntanglement"]):
    """`BellStateEntanglement` is the ideal max entangled qubits. Its fidelity is always 1."""

    @property
    @override
    def fidelity(self) -> float:
        return 1.0

    @fidelity.setter
    @override
    def fidelity(self, value: float):
        assert value == 1.0, "BellStateEntanglement fidelity is always 1"

    @override
    def swapping(self, epr: "BellStateEntanglement", *, name: str | None = None, ps: float = 1) -> "BellStateEntanglement|None":
        _ = ps
        ne = BellStateEntanglement(name=name)
        if self.is_decoherenced or epr.is_decoherenced:
            return None
        epr.is_decoherenced = True
        self.is_decoherenced = True
        return ne

    @override
    def distillation(self, epr: "BellStateEntanglement") -> "BellStateEntanglement":
        ne = BellStateEntanglement()
        if self.is_decoherenced or epr.is_decoherenced:
            ne.is_decoherenced = True
            ne.fidelity = 0
        epr.is_decoherenced = True
        self.is_decoherenced = True
        return ne
