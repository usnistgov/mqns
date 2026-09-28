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

from mqns.models.epr.bell import BellStateEntanglement
from mqns.models.epr.entanglement import Entanglement, EntanglementInitKwargs, PurifProtocol
from mqns.models.epr.input import EPR_TYPE_MAP, EprTypeInput, EprTypeLiteral, parse_epr_type
from mqns.models.epr.mixed import MixedStateEntanglement
from mqns.models.epr.qubit_pair import EntangledQubitPair
from mqns.models.epr.werner import WernerStateEntanglement

__all__ = [
    "BellStateEntanglement",
    "EntangledQubitPair",
    "Entanglement",
    "EntanglementInitKwargs",
    "EPR_TYPE_MAP",
    "EprTypeInput",
    "EprTypeLiteral",
    "MixedStateEntanglement",
    "parse_epr_type",
    "PurifProtocol",
    "WernerStateEntanglement",
]

for name in __all__:
    if name in (
        "EPR_TYPE_MAP",
        "EprTypeInput",
        "EprTypeLiteral",
    ):
        continue
    globals()[name].__module__ = __name__
