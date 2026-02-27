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

from typing import override

from mqns.entity.memory import MemoryQubit
from mqns.network.fw import Forwarder
from mqns.network.network import TimingPhase, TimingPhaseEvent
from mqns.network.protocol.event import ManageActiveChannels
from mqns.network.reactive.message import LinkStateEntry, LinkStateMsg
from mqns.utils import log


class ReactiveForwarder(Forwarder):
    """
    ReactiveForwarder is the forwarder of QNodes. It continuously generates EPRs on all links,
    then sends EPR link state to the controller to receive routing instructions.
    Works only in Synchronous timing mode.
    It implements the forwarding phase (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller.
    """

    @override
    def install(self, node):
        super().install(node)

        # Qchannel activation is called before simulation starts, but EPR generation starts at t=0
        self.activate_qchannels()

    def activate_qchannels(self):
        """
        Instruct LinkLayer to start generating EPRs on ALL qchannels.
        This may be called from install() or at the first EXTERNAL phase (for better coordination).
        """
        for ch in self.node.qchannels:
            if self.node == ch.node_list[0]:  # self is the EPR initiator node for this channel
                log.debug(f"{self.node}: activate qchannel {ch.name}")
                self.simulator.add_event(
                    ManageActiveChannels(
                        self.node,
                        ch.node_list[1],
                        ch,
                        path_id=None,
                        start=True,
                        t=self.simulator.tc,
                        by=self,
                    )
                )

    @override
    def handle_sync_phase(self, event: TimingPhaseEvent):
        """
        Handle timing phase signals, only used in SYNC timing mode.

        Upon entering ROUTING phase:

        1. Send to controller link states corresponding to entangled qubits that arrived during EXTERNAL phase
           and wait for routing instructions.
        """
        if event.phase == TimingPhase.ROUTING:
            log.debug(f"{self.node}: there are {len(self.waiting_etg)} etg qubits to process")
            log.debug(f"{self.node}: send link_state for {len(self.waiting_etg)} etg qubits")
            self.send_link_state()
        else:
            super().handle_sync_phase(event)

    @override
    def handle_path_change(self, **_):
        """
        Process LinkLayer changes after a path has been installed or uninstalled.

        This does nothing because LinkLayer is always running based on topology.
        """

    def send_link_state(self):
        """
        Send link state message to controller. Assumes direct connection to controller.
        """
        link_states: list[LinkStateEntry] = []
        for event in self.waiting_etg:
            link_states.append({"node": event.node.name, "neighbor": event.neighbor.name, "qubit": event.qubit.addr})

        if len(link_states) == 0:
            log.debug(f"{self.node}: no link_state to send")
            return

        msg: LinkStateMsg = {
            "cmd": "LS",
            "ls": link_states,
        }
        self.send_ctrl(msg)

    @override
    def release_qubit(self, qubit: MemoryQubit, *, need_remove=False):
        """
        Release a qubit.

        Args:
            need_remove: whether to remove the data associated with the qubit.
                         This should be set to True unless .read(remove=True) is already performed.
        """
        super().release_qubit(qubit, need_remove=need_remove)

        # in Reactive: remove path allocation for next routing cycle
        self.memory.deallocate(qubit.addr)
