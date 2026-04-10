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

import copy
from abc import abstractmethod
from typing import TypedDict, Unpack, override

import numpy as np

from mqns.entity.memory import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.fw.cutoff import CutoffScheme, CutoffSchemeWaitTime
from mqns.network.fw.fib import Fib, FibEntry
from mqns.network.fw.fw_classic import ForwarderClassicMixin, fw_control_cmd_handler, fw_signaling_cmd_handler
from mqns.network.fw.fw_purif import ForwarderPurifProc
from mqns.network.fw.fw_swap import ForwarderSwapProc
from mqns.network.fw.message import (
    CutoffDiscardMsg,
    InstallPathMsg,
    PurifResponseMsg,
    PurifSolicitMsg,
    SwapUpdateMsg,
    UninstallPathMsg,
)
from mqns.network.fw.mux import MuxScheme
from mqns.network.fw.mux_buffer_space import MuxSchemeBufferSpace
from mqns.network.fw.select import SelectPurifQubit, call_select_purif_qubit
from mqns.network.network import TimingPhase, TimingPhaseEvent
from mqns.network.protocol.event import QubitEntangledEvent, QubitReleasedEvent
from mqns.utils import json_encodable, log


class ForwarderInitKwargs(TypedDict, total=False):
    ps: float
    """Probability of successful entanglement swapping, default is 1.0."""
    cutoff: CutoffScheme | None
    """EPR age cut-off scheme, default is wait-time."""
    mux: MuxScheme | None
    """Path multiplexing scheme, default is buffer-space."""
    select_purif_qubit: SelectPurifQubit
    """Qubit selection among purification candidates, default is picking first candidate."""


@json_encodable
class ForwarderCounters:
    """Counters of ``Forwarder``."""

    def __init__(self):
        self.n_entg = 0
        """How many elementary entanglements received from link layer."""
        self.n_purif: list[int] = []
        """How many entanglements completed i-th purif round (zero-based index)."""
        self.n_eligible = 0
        """How many entanglements completed all purif rounds and became eligible."""
        self.n_swapped_s = 0
        """How many swaps succeeded sequentially."""
        self.n_swapped_p = 0
        """How many swaps succeeded with parallel merging."""
        self.n_swap_conflict = 0
        """How many swaps were skipped due to conflictual decisions."""
        self.n_consumed = 0
        """How many entanglements were consumed (either end-to-end or in swap-disabled mode)."""
        self.consumed_sum_fidelity = 0.0
        """Sum of fidelity of consumed entanglement.s"""
        self.consumed_fidelity_values: list[float] | None = None
        """Fidelity values of consumed entanglements, None disables collection."""
        self.n_cutoff = [0, 0]
        """
        How many entanglements are discarded by CutoffScheme.

        * [0]: swap_cutoff exceeded locally
        * [1]: swap_cutoff exceeded on partner forwarder
        * [2r+0]: purif_cutoff[r] exceeded locally
        * [2r+1]: purif_cutoff[r] exceeded on partner forwarder
        """

    def enable_collect_all(self) -> None:
        """Enable collecting all values for histogram generation."""
        assert self.n_consumed == 0
        self.consumed_fidelity_values = []

    def increment_n_purif(self, i: int) -> None:
        if len(self.n_purif) <= i:
            self.n_purif += [0] * (i + 1 - len(self.n_purif))
        self.n_purif[i] += 1

    def increment_n_consumed(self, fidelity: float) -> None:
        self.n_consumed += 1
        self.consumed_sum_fidelity += fidelity
        if self.consumed_fidelity_values is not None:
            self.consumed_fidelity_values.append(fidelity)

    def increment_n_cutoff(self, round: int, local: bool) -> None:
        minlen = 2 * (round + 1)
        if len(self.n_cutoff) < minlen:
            self.n_cutoff += [0] * (minlen - len(self.n_cutoff))
        self.n_cutoff[2 * round + (0 if local else 1)] += 1

    @property
    def n_swapped(self) -> int:
        """How many swaps succeeded."""
        return self.n_swapped_s + self.n_swapped_p

    @property
    def consumed_avg_fidelity(self) -> float:
        """Average fidelity of consumed entanglements."""
        if self.n_consumed == 0:
            return 0.0
        if self.consumed_fidelity_values is None:
            return self.consumed_sum_fidelity / self.n_consumed
        return np.mean(self.consumed_fidelity_values).item()

    def __repr__(self) -> str:
        return (
            f"entg={self.n_entg} purif={self.n_purif} eligible={self.n_eligible} "
            f"swapped={self.n_swapped_s}+{self.n_swapped_p} "
            f"swap-conflict={self.n_swap_conflict} cutoff-discard={self.n_cutoff} "
            f"consumed={self.n_consumed} (F={self.consumed_avg_fidelity})"
        )


class Forwarder(ForwarderClassicMixin, Application[QNode]):
    """
    Forwarder is the network layer component of QNodes implementing the forwarding phase
    (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller.
    """

    def __init__(self, **kwargs: Unpack[ForwarderInitKwargs]):
        """
        This constructor sets up a node's entanglement forwarding logic in a quantum network.
        It configures the swapping success probability and preparing internal
        state for managing memory, routing instructions (via FIB), synchronization,
        and classical communication handling.
        """
        super().__init__()
        self._init_classic_mixin()

        self.cutoff: CutoffScheme = copy.deepcopy(kwargs.get("cutoff")) or CutoffSchemeWaitTime()
        """EPR age cut-off scheme."""
        self.mux: MuxScheme = copy.deepcopy(kwargs.get("mux")) or MuxSchemeBufferSpace()
        """Multiplexing scheme."""
        self._select_purif_qubit = kwargs.get("select_purif_qubit")

        self.fib = Fib()
        """FIB structure."""
        self.purif = ForwarderPurifProc()
        self.swap = ForwarderSwapProc(ps=kwargs.get("ps", 1.0))

        self.add_handler(self.handle_sync_phase, TimingPhaseEvent)
        self.add_handler(self.qubit_is_entangled, QubitEntangledEvent)

        self.waiting_etg: list[QubitEntangledEvent] = []
        """
        Elementary-entangled qubits received during EXTERNAL phase.
        These are buffered until INTERNAL phase starts.
        """

        self.cnt = ForwarderCounters()
        """
        Counters.
        """

    @override
    def install(self, node):
        self._application_install(node, QNode)
        self.memory = self.node.memory
        """Quantum memory of the node."""
        self.network = self.node.network
        """Quantum network that contains the node."""
        self.epr_type = self.network.epr_type
        """Network-wide entanglement type."""

        self.cutoff.install(self)
        self.mux.install(self)
        self.purif.install(self)
        self.swap.install(self)

    def handle_sync_phase(self, event: TimingPhaseEvent):
        """
        Handle timing phase signals, only used in SYNC timing mode.

        Upon entering INTERNAL phase:

        1. Start processing elementary entanglements that arrived during EXTERNAL phase.

        Upon exiting INTERNAL phase:

        1. Clear ``remote_swapped_eprs``.
           All memory qubits are being discarded by LinkLayer, so that these have become useless.
        """
        match event.action:
            case TimingPhase.INTERNAL, True:
                log.debug(f"{self.node}: there are {len(self.waiting_etg)} etg qubits to process")
                for etg_event in self.waiting_etg:
                    self.qubit_is_entangled(etg_event)
                self.waiting_etg.clear()
            case TimingPhase.INTERNAL, False:
                self.swap.remote_swapped_eprs.clear()

    @fw_control_cmd_handler("INSTALL_PATH")
    def handle_install_path(self, msg: InstallPathMsg):
        """
        Process an INSTALL_PATH message from the controller.

        1. Insert FIB entry.
        2. Identify neighbors and qchannels.
        3. Save the path and neighbors in the multiplexing scheme.
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
        # associate path with qchannel and allocate qubits
        if l_neighbor := self._find_neighbor(fib_entry, -1):
            self.mux.install_path_neighbor(instructions, fib_entry, PathDirection.L, *l_neighbor)
        if r_neighbor := self._find_neighbor(fib_entry, +1):
            self.mux.install_path_neighbor(instructions, fib_entry, PathDirection.R, *r_neighbor)

        # call subclass specialization
        self.handle_path_change(
            path_id=path_id,
            uninstall=False,
            fib_entry=fib_entry,
            l_neighbor=l_neighbor,
            r_neighbor=r_neighbor,
        )

    @fw_control_cmd_handler("UNINSTALL_PATH")
    def handle_uninstall_path(self, msg: UninstallPathMsg):
        """
        Process an UNINSTALL_PATH message from the controller.

        1. Insert FIB entry.
        2. Identify neighbors and qchannels.
        3. Save the path and neighbors in the multiplexing scheme.
        4. Notify LinkLayer to start elementary EPR generation toward the right neighbor.
        """
        path_id = msg["path_id"]

        # retrieve and erase FIB entry
        fib_entry = self.fib.get(path_id)
        self.fib.erase(path_id)

        # identify left/right neighbors
        # disassociate path with qchannel and deallocate qubits
        if l_neighbor := self._find_neighbor(fib_entry, -1):
            self.mux.uninstall_path_neighbor(fib_entry, PathDirection.L, *l_neighbor)
        if r_neighbor := self._find_neighbor(fib_entry, +1):
            self.mux.uninstall_path_neighbor(fib_entry, PathDirection.R, *r_neighbor)

        # call subclass specialization
        self.handle_path_change(
            path_id=path_id,
            uninstall=True,
            fib_entry=fib_entry,
            l_neighbor=l_neighbor,
            r_neighbor=r_neighbor,
        )

    def _find_neighbor(self, fib_entry: FibEntry, route_offset: int) -> tuple[QNode, QuantumChannel] | None:
        neigh_idx = fib_entry.own_idx + route_offset
        if neigh_idx in (-1, len(fib_entry.route)):  # no left/right neighbor if own node is the left/right end node
            return None
        neigh = self.network.get_node(fib_entry.route[neigh_idx])
        return neigh, self.node.get_qchannel(neigh)

    @abstractmethod
    def handle_path_change(
        self,
        *,
        path_id: int,
        uninstall: bool,
        fib_entry: FibEntry,
        l_neighbor: tuple[QNode, QuantumChannel] | None,
        r_neighbor: tuple[QNode, QuantumChannel] | None,
    ):
        """
        Process LinkLayer changes after a path has been installed or uninstalled.

        Args:
            path_id: Path identifier.
            uninstall: Whether this is an uninstall command.
            fib_entry: FIB entry.
            l_neighbor: Left neighbor and channel toward it.
            r_neighbor: Right neighbor and channel toward it.
        """

    @fw_signaling_cmd_handler("CUTOFF_DISCARD")
    def _handle_cutoff_discard(self, msg: CutoffDiscardMsg, fib_entry: FibEntry):
        _ = fib_entry
        self.cutoff.handle_discard(msg)

    @fw_signaling_cmd_handler("PURIF_SOLICIT")
    def _handle_purif_solicit(self, msg: PurifSolicitMsg, fib_entry: FibEntry):
        self.purif.handle_solicit(msg, fib_entry)

    @fw_signaling_cmd_handler("PURIF_RESPONSE")
    def _handle_purif_response(self, msg: PurifResponseMsg, fib_entry: FibEntry):
        self.purif.handle_response(msg, fib_entry)

    @fw_signaling_cmd_handler("SWAP_UPDATE")
    def _handle_swap_update(self, msg: SwapUpdateMsg, fib_entry: FibEntry):
        self.swap.handle_update(msg, fib_entry)

    def qubit_is_entangled(self, event: QubitEntangledEvent):
        """
        Handle a qubit entering ENTANGLED state, i.e. having an elementary entanglement.

        In ASYNC timing mode, events are processed immediately.
        In SYNC timing mode, events arrive in EXTERNAL phase and is queued in ``self.waiting_etg``.
        Queued events are released upon entering INTERNAL phase and then processed.

        The actual processing is handled by the multiplexing scheme.

        If a SwapUpdate was received before processing this event and buffered in ``self.waiting_su``,
        it is re-processed at this time.

        Args:
            event: Event containing the entangled qubit and its associated metadata (e.g., neighbor).

        """
        if not self.node.timing.is_internal():  # in SYNC timing mode EXTERNAL phase
            self.waiting_etg.append(event)
            return

        self.cnt.n_entg += 1

        qubit = event.qubit
        assert qubit.state == QubitState.ENTANGLED1
        _, epr = self.memory.read(qubit.addr, has=self.epr_type)
        log.debug(f"{self.node}: ENTANGLED {qubit} | {epr}")
        self.mux.qubit_is_entangled(qubit, epr, event.neighbor)

        self.swap.pop_waiting_su(qubit)

    def qubit_is_purif(self, qubit: MemoryQubit, fib_entry: FibEntry, partner: QNode):
        """
        Handle a qubit entering PURIF state or have completed a previous purification round.

        1. Determines the segment in which the qubit is entangled and number of required purification rounds.
        2. If the required rounds are completed, the qubit becomes eligible.
        3. Otherwise, check if own node is primary for the purification protocol.
           If so, search for an auxiliary qubit to use, release the auxiliary qubit,
           and send PURIF_SOLICIT to the partner node.

        Args:
            qubit: The memory qubit at PURIF state.
            fib_entry: FIB entry containing routing and purification instructions.
            partner: The node with which the qubit shares an EPR.
        """
        assert qubit.state == QubitState.PURIF
        assert qubit.qchannel is not None

        own_idx, own_rank = fib_entry.own_idx, fib_entry.own_swap_rank
        partner_idx, partner_rank = fib_entry.find_index_and_swap_rank(partner.name)
        if own_rank > partner_rank:
            # swapping order disallows initiating purif / swap / consumption
            return

        segment_name = f"{self.node.name}-{partner.name}" if own_idx < partner_idx else f"{partner.name}-{self.node.name}"
        want_rounds = fib_entry.purif.get(segment_name, 0)
        log.debug(
            f"{self.node}: segment {segment_name} (qubit {qubit.addr}) has "
            + f"{qubit.purif_rounds} and needs {want_rounds} purif rounds"
        )

        if qubit.purif_rounds == want_rounds:
            self.cnt.n_eligible += 1
            qubit.purif_rounds = 0
            qubit.state = QubitState.ELIGIBLE
            self.qubit_is_eligible(qubit, fib_entry)
            return
        assert qubit.purif_rounds < want_rounds

        is_primary = (own_rank, own_idx) < (partner_rank, partner_idx)
        if not is_primary:
            log.debug(f"{self.node}: is not primary node for segment {segment_name} purif")
            return

        candidates = self.memory.find(
            lambda q, v: (
                q.addr != qubit.addr  # not the same qubit
                and q.state == QubitState.PURIF  # in PURIF state
                and q.purif_rounds == qubit.purif_rounds  # with same number of purif rounds
                and partner in (v.src, v.dst)  # with the same partner
                and q.path_id == fib_entry.path_id  # on the same path_id
            ),
            has=self.epr_type,
        )
        found = call_select_purif_qubit(self._select_purif_qubit, qubit, fib_entry, partner, candidates)
        if not found:
            log.debug(f"{self.node}: no candidate EPR for segment {segment_name} purif round {1 + qubit.purif_rounds}")
            return

        self.purif.start(qubit, found[0], fib_entry, partner)

    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FibEntry | None):
        """
        Handle a qubit entering ELIGIBLE state.

        If this is an end node of the path, consume the EPR.

        Otherwise, update the EPR age cut-off scheme, and then attempt entanglement swapping:

        1. Look for a matching eligible qubit to perform swapping.
        2. Generate a new EPR if successful.
        3. Notify adjacent nodes with SWAP_UPDATE messages.

        Args:
            qubit: The qubit that became eligible.
            fib_entry: FIB entry (not available with MuxSchemeStatistical).
        """
        assert qubit.state == QubitState.ELIGIBLE
        if not self.node.timing.is_internal():
            log.debug(f"{self.node}: INT phase is over -> stop swaps")
            return

        _, epr = self.memory.read(qubit.addr, has=self.epr_type)
        if self.can_consume(fib_entry, epr):
            self.consume_and_release(qubit)
            return

        self.cutoff.qubit_is_eligible(qubit, fib_entry)

        swap_candidates = self.memory.find(
            lambda q, _: (
                q.state == QubitState.ELIGIBLE  # in ELIGIBLE state
                and q.qchannel != qubit.qchannel  # assigned to a different channel
                and self.cutoff.filter_swap_candidate(q)
            ),
            has=self.epr_type,
        )
        swap_candidate_tuple = self.mux.find_swap_candidate(qubit, epr, fib_entry, swap_candidates)
        mq1: MemoryQubit | None = None
        if swap_candidate_tuple:
            mq1, fib_entry = swap_candidate_tuple
            self.swap.start(qubit, mq1, fib_entry)
        self.cutoff.before_swap(qubit, mq1, fib_entry)

    def can_consume(self, fib_entry: FibEntry | None, epr: Entanglement) -> bool:
        if fib_entry is None:
            assert epr.src is not None
            assert epr.dst is not None
            src, dst = epr.src.name, epr.dst.name
            return next(self.fib.find_request(lambda g: g.src == src and g.dst == dst), None) is not None

        return fib_entry.is_swap_disabled or fib_entry.own_idx in (0, len(fib_entry.route) - 1)

    def consume_and_release(self, qubit: MemoryQubit):
        """
        Consume an entangled qubit.
        """
        _, qm = self.memory.read(qubit.addr, has=self.epr_type, set_fidelity=True, remove=True)
        log.debug(f"{self.node}: consume EPR: {qm}")
        self.cnt.increment_n_consumed(qm.fidelity)

        self.release_qubit(qubit)

    def release_qubit(self, qubit: MemoryQubit, *, need_remove=False):
        """
        Release a qubit.

        Args:
            need_remove: whether to remove the data associated with the qubit.
                         This should be set to True unless .read(remove=True) is already performed.
        """
        if need_remove:
            self.memory.read(qubit.addr, remove=True)

        qubit.state = QubitState.RELEASE
        self.simulator.add_event(QubitReleasedEvent(self.node, qubit, t=self.simulator.tc))
