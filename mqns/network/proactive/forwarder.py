#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
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

from typing import Final, NotRequired, Unpack, cast, override

from mqns.entity.memory import MemoryQubit
from mqns.models.epr import Entanglement
from mqns.network.fw import FibPath, FibRequest, Forwarder, ForwarderInitKwargs
from mqns.network.proactive.fw_nb import ProactiveForwarderNorthbound
from mqns.network.proactive.mux import MuxScheme
from mqns.network.proactive.mux_input import MuxSchemeInput, parse_mux_scheme


class ProactiveForwarderInitKwargs(ForwarderInitKwargs):
    mux: NotRequired[MuxSchemeInput]
    """Path multiplexing scheme, default is buffer-space."""


class ProactiveForwarder(Forwarder):
    """
    ProactiveForwarder is the forwarder of QNodes and receives routing instructions from the controller.
    It implements the forwarding phase (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller.
    """

    nb: Final[ProactiveForwarderNorthbound]
    """Northbound interface to communicate with the ProactiveRoutingController."""

    mux: Final[MuxScheme]
    """Multiplexing scheme."""

    def __init__(self, **kwargs: Unpack[ProactiveForwarderInitKwargs]):
        mux = parse_mux_scheme(kwargs.pop("mux", None))
        super().__init__(**cast(ForwarderInitKwargs, kwargs))
        self.nb = ProactiveForwarderNorthbound()
        self.mux = mux

    @override
    def install(self, node):
        super().install(node)
        self.nb.install(self)
        self.mux.install(self)

    @override
    def qubit_is_entangled_next(self, mq: MemoryQubit, epr: Entanglement) -> FibPath | None:
        return self.mux.qubit_is_entangled(mq, epr)

    @override
    def find_swap_with(self, mq0: MemoryQubit, epr0: Entanglement, fp: FibPath | None) -> tuple[MemoryQubit, FibPath] | None:
        return self.mux.find_swap_with(mq0, epr0, fp)

    @override
    def request_reached_epr_count(self, fr: FibRequest) -> None:
        self.nb.send_reach_epr_count(fr)
