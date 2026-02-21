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

from mqns.entity.memory import MemoryQubit, PathDirection
from mqns.network.fw import Forwarder, fw_control_cmd_handler
from mqns.network.fw.fib import FibEntry
from mqns.network.fw.message import InstallPathMsg, UninstallPathMsg
from mqns.network.network import TimingPhase, TimingPhaseEvent
from mqns.network.proactive.cutoff import CutoffScheme, CutoffSchemeWaitTime
from mqns.network.proactive.mux import MuxScheme
from mqns.network.proactive.mux_buffer_space import MuxSchemeBufferSpace
from mqns.network.proactive.select import SelectPurifQubit
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

    def __init__(
        self,
        *,
        ps: float = 1.0,
        cutoff: CutoffScheme = CutoffSchemeWaitTime(),
        mux: MuxScheme = MuxSchemeBufferSpace(),
        select_purif_qubit: SelectPurifQubit = None,
    ):
        """
        This constructor sets up a node's entanglement forwarding logic in a quantum network.
        It configures the swapping success probability and preparing internal
        state for managing memory, routing instructions (via FIB), synchronization,
        and classical communication handling.

        Args:
            ps: Probability of successful entanglement swapping (default: 1.0).
            mux: Path multiplexing scheme (default: buffer-space).
        """
        super().__init__(ps=ps, cutoff=cutoff, mux=mux, select_purif_qubit=select_purif_qubit)

    @override
    def install(self, node):
        super().install(node)

        # Qchannel activation is called before simulation starts, but EPR generation starts at t=0
        self.activate_qchannels()

    # instruct LinkLayer to start generating EPRs on ALL qchannels
    # can be called from install() or at the first EXTERNAL phase (for better coordination)
    def activate_qchannels(self):
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

        Upon entering EXTERNAL phase:

        1. Clear `remote_swapped_eprs`.
           All memory qubits are being discarded by LinkLayer, so that these have become useless.

        Upon entering INTERNAL phase:

        1. Send to controller link states corresponding to entangled qubits that arrived during EXTERNAL phase
        and wait for routing instructions.
        2. Start processing elementary entanglements that arrived during EXTERNAL phase following routing instructions.
        """
        if event.phase == TimingPhase.EXTERNAL:
            self.remote_swapped_eprs.clear()
        elif event.phase == TimingPhase.ROUTING:
            log.debug(f"{self.node}: there are {len(self.waiting_etg)} etg qubits to process")
            log.debug(f"{self.node}: send link_state for {len(self.waiting_etg)} etg qubits")
            self.send_link_state()
        elif event.phase == TimingPhase.INTERNAL:
            log.debug(f"{self.node}: there are {len(self.waiting_etg)} etg qubits to process")
            for etg_event in self.waiting_etg:
                self.qubit_is_entangled(etg_event)
            self.waiting_etg.clear()

    @fw_control_cmd_handler("install_path")
    def handle_install_path(self, msg: InstallPathMsg):
        """
        Process an install_path message containing routing instructions from the controller.

        1. Insert FIB entry -> needed for signaling
        2. Save the path and neighbors in the multiplexing scheme -> to reuse forwarding logic
        """
        path_id = msg["path_id"]
        instructions = msg["instructions"]
        self.mux.validate_path_instructions(instructions)

        # populate FIB
        route = instructions["route"]
        fib_entry = FibEntry(
            path_id=path_id,
            req_id=instructions["req_id"],
            route=route,
            own_idx=route.index(self.node.name),
            swap=instructions["swap"],
            swap_cutoff=[None if t < 0 else self.simulator.time(time_slot=t) for t in instructions["swap_cutoff"]],
            purif=instructions["purif"],
        )
        self.fib.insert_or_replace(fib_entry)

        # identify left/right neighbors
        l_neighbor = self._find_neighbor(fib_entry, -1)
        r_neighbor = self._find_neighbor(fib_entry, +1)

        if l_neighbor:
            l_qchannel = self.node.get_qchannel(l_neighbor)

            # associate path with qchannel and allocate qubits
            self.mux.install_path_neighbor(instructions, fib_entry, PathDirection.L, l_neighbor, l_qchannel)

        if r_neighbor:
            r_qchannel = self.node.get_qchannel(r_neighbor)

            # associate path with qchannel and allocate qubits
            self.mux.install_path_neighbor(instructions, fib_entry, PathDirection.R, r_neighbor, r_qchannel)

    @fw_control_cmd_handler("uninstall_path")
    def handle_uninstall_path(self, msg: UninstallPathMsg):
        raise NotImplementedError

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
