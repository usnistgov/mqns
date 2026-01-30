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

from typing import Unpack, final, override

from mqns.models.epr.entanglement import Entanglement, EntanglementInitKwargs


@final
class BellStateEntanglement(Entanglement):
    """`BellStateEntanglement` is the ideal max entangled qubits. Its fidelity is always 1."""

    @property
    @override
    def fidelity(self) -> float:
        return 1.0

    @fidelity.setter
    @override
    def fidelity(self, value: float):
        assert value == 1.0, "BellStateEntanglement fidelity is always 1"

    @staticmethod
    @override
    def _make_swapped(epr0: "BellStateEntanglement", epr1: "BellStateEntanglement", **kwargs: Unpack[EntanglementInitKwargs]):
        _ = epr0, epr1
        return BellStateEntanglement(**kwargs)

    @override
    def _do_purify(self, epr1: "BellStateEntanglement") -> bool:
        _ = epr1
        return True

    @override
    def apply_error(self, error) -> None:
        _ = error
