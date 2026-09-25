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

from typing import Final, Unpack, override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.models.epr import Entanglement
from mqns.network.fw import FibPath, Forwarder, ForwarderInitKwargs
from mqns.network.network import TimingPhase, sync_phase_handler
from mqns.network.protocol.event import PathActivateEvent, QubitEntangledEvent
from mqns.network.reactive.fw_nb import ReactiveForwarderNorthbound
from mqns.network.reactive.fw_plan import ReactivePlanner


class ReactiveForwarder(Forwarder):
    """
    ReactiveForwarder is the forwarder of QNodes. It continuously generates EPRs on all links,
    then sends EPR link state to the controller to receive routing instructions.
    Works only in Synchronous timing mode.
    It implements the forwarding phase (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller.
    """

    planner: Final[ReactivePlanner]
    """Store the planned action for each qubit."""

    nb: Final[ReactiveForwarderNorthbound]
    """Northbound interface to communicate with the ReactiveRoutingController."""

    def __init__(self, **kwargs: Unpack[ForwarderInitKwargs]):
        super().__init__(**kwargs)
        self.planner = ReactivePlanner()
        self.nb = ReactiveForwarderNorthbound(self.planner)

    @override
    def install(self, node):
        super().install(node)
        self.nb.install(self)

        # Qchannel activation is called before simulation starts, but EPR generation starts at t=0
        self.activate_qchannels()

    def activate_qchannels(self):
        for ch in self.node.qchannels:
            is_primary = ch.node_list[0] is self.node
            self.simulator.sched(PathActivateEvent(self.node, ch, None, t=self.simulator.tc, is_primary=is_primary))

    @sync_phase_handler(TimingPhase.ROUTING, True)
    def sync_routing_enter(self):
        # Transmit link states based on entangled qubits arrived during EXTERNAL phase.
        self.nb.send_link_state()

    @sync_phase_handler(TimingPhase.INTERNAL, False)
    def sync_internal_exit(self):
        # Clear FIB and path assignments, as these are only useful for one slot.
        self.fib.clear()
        self.planner.clear()

    @override
    def qubit_is_entangled_in_external(self, event: QubitEntangledEvent) -> None:
        """
        Handle a qubit entering ENTANGLED state when in EXTERNAL phase of SYNC timing mode.
        """
        super().qubit_is_entangled_in_external(event)
        self.nb.append_link_state(event.neighbor, event.qubit)

    @override
    def qubit_is_entangled_next(self, mq: MemoryQubit, epr: Entanglement) -> FibPath | None:
        _ = epr
        fp = self.planner.find_fib_path(mq)
        if fp is None:
            return None

        mq.epr_path_ids = [fp.path_id]
        mq.state = QubitState.PURIF
        return fp

    @override
    def find_swap_with(self, mq0: MemoryQubit, epr0: Entanglement, fp: FibPath | None) -> tuple[MemoryQubit, FibPath] | None:
        _ = epr0, fp
        return self.planner.find_swap_with(mq0)
