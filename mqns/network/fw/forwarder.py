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
from collections.abc import Callable
from typing import Final, Literal, TypedDict, Unpack, override

from mqns.entity.cchannel import ClassicCommandDispatcherMixin
from mqns.entity.memory import MemoryDecohereEvent, MemoryQubit, PathDirection, QubitState
from mqns.entity.node import Application, QNode
from mqns.models.delay import DelayInput, parse_delay
from mqns.models.epr import Entanglement
from mqns.models.error import PerfectErrorModel
from mqns.models.error.input import ErrorModelInputBasic, parse_error
from mqns.network.fw.cutoff import CutoffScheme, CutoffSchemeWaitTime
from mqns.network.fw.fib import Fib, FibPath, FibRequest
from mqns.network.fw.fw_purif import ForwarderPurifProc
from mqns.network.fw.fw_swap import ForwarderSwapProc
from mqns.network.fw.select import MemoryEprTuple, call_select
from mqns.network.network import TimingPhase, sync_phase_handler
from mqns.network.protocol.event import QubitConsumeEvent, QubitEntangledEvent, QubitReleasedEvent
from mqns.simulator import event_handler
from mqns.utils import json_encodable, unwrap, unwrap_cast


class ForwarderInitKwargs(TypedDict, total=False):
    p_swap: float
    """Probability of successful entanglement swapping, default is ``1.0``."""
    swap_delay: DelayInput
    """Swapping delay model, default is zero."""
    swap_error: ErrorModelInputBasic
    """Swapping error model, default is perfect."""
    swap_error_at: Literal["s", "f"]
    """Swapping error applied at start or finish time, default is ``s``."""
    cutoff: CutoffScheme | None
    """EPR age cut-off scheme, default is wait-time."""
    select_purif_qubit: Callable[[list[MemoryEprTuple], MemoryQubit, FibPath, QNode], MemoryEprTuple] | None
    """Qubit selection among purification candidates, default is picking first candidate."""


@json_encodable
class ForwarderCounters:
    """Counters of ``Forwarder``."""

    def __init__(self):
        super().__init__()

        self.n_entg = 0
        """How many elementary entanglements received from link layer."""
        self.n_purif: list[int] = []
        """How many entanglements completed i-th purif round (zero-based index)."""
        self.n_eligible = 0
        """How many entanglements completed all purif rounds and became eligible."""
        self.n_swapped = 0
        """How many physical swaps succeeded."""
        self.n_swap_fail = 0
        """How many physical swaps failed."""
        self.n_su_lower = [0, 0, 0, 0, 0]
        """
        How many SWAP_UPDATE messages from lower-ranked node were processed.

        * [0]: normal
        * [1]: qubit decohered
        * [2]: lower-expiry
        * [3]: lower-swap-failure
        * [4]: lower-swap-conflict
        """
        self.n_su_same = [0, 0]
        """
        How many SWAP_UPDATE messages from same-ranked node were processed.

        * [0]: normal
        * [1]: previous-swap-failure
        """
        self.n_cutoff = [0, 0]
        """
        How many entanglements are discarded by CutoffScheme.

        * [0]: swap_cutoff exceeded locally
        * [1]: swap_cutoff exceeded on partner forwarder
        * [2r+0]: purif_cutoff[r] exceeded locally
        * [2r+1]: purif_cutoff[r] exceeded on partner forwarder
        """

    def increment_n_purif(self, i: int) -> None:
        if len(self.n_purif) <= i:
            self.n_purif += [0] * (i + 1 - len(self.n_purif))
        self.n_purif[i] += 1

    def increment_n_cutoff(self, round: int, local: bool) -> None:
        minlen = 2 * (round + 1)
        if len(self.n_cutoff) < minlen:
            self.n_cutoff += [0] * (minlen - len(self.n_cutoff))
        self.n_cutoff[2 * round + (0 if local else 1)] += 1

    def __repr__(self) -> str:
        return (
            f"entg={self.n_entg} purif={self.n_purif} eligible={self.n_eligible} "
            f"swapped={self.n_swapped} swap-fail={self.n_swap_fail} "
            f"su-lower={self.n_su_lower} su-same={self.n_su_same} cutoff-discard={self.n_cutoff}"
        )


class Forwarder(ClassicCommandDispatcherMixin, Application[QNode]):
    """
    Forwarder is the network layer component of QNodes implementing the forwarding phase
    (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller.
    """

    cutoff: Final[CutoffScheme]
    """EPR age cut-off scheme."""

    fib: Final[Fib]
    """FIB data structure."""

    purif: Final[ForwarderPurifProc]
    """Purification procedure module."""

    swap: Final[ForwarderSwapProc]
    """Swapping procedure module."""

    cnt: Final[ForwarderCounters]
    """Counters."""

    def __init__(self, **kwargs: Unpack[ForwarderInitKwargs]):
        """
        This constructor sets up a node's entanglement forwarding logic in a quantum network.
        It configures the swapping success probability and preparing internal
        state for managing memory, routing instructions (via FIB), synchronization,
        and classical communication handling.
        """
        super().__init__()

        self.cutoff = copy.deepcopy(kwargs.get("cutoff")) or CutoffSchemeWaitTime()
        self._select_purif_qubit = kwargs.get("select_purif_qubit")

        self.fib = Fib()
        self.purif = ForwarderPurifProc()
        self.swap = ForwarderSwapProc(
            ps=kwargs.get("p_swap", 1.0),
            delay=parse_delay(kwargs.get("swap_delay", 0)),
            error=parse_error(kwargs.get("swap_error"), PerfectErrorModel, -1),
            error_at_finish=kwargs.get("swap_error_at", "s") == "f",
        )

        self.entangled_in_external: list[QubitEntangledEvent] = []
        """
        Elementary entanglements received during EXTERNAL phase.
        These are buffered until INTERNAL phase starts.
        """

        self.cnt = ForwarderCounters()

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
        self.fib.install(self)
        self.purif.install(self)
        self.swap.install(self)

    @sync_phase_handler(TimingPhase.INTERNAL, True)
    def sync_internal_enter(self) -> None:
        # Start processing elementary entanglements that arrived during EXTERNAL phase.
        for etg_event in self.entangled_in_external:
            self.qubit_is_entangled_in_internal(etg_event)
        self.entangled_in_external.clear()

    @sync_phase_handler(TimingPhase.INTERNAL, False)
    def sync_internal_exit(self) -> None:
        self.swap.sync_internal_exit()

    @event_handler
    def qubit_is_entangled(self, event: QubitEntangledEvent) -> None:
        """
        Handle a qubit entering ENTANGLED state, i.e. having an elementary entanglement.

        In ASYNC timing mode, events are processed immediately.
        In SYNC timing mode, events arrive in EXTERNAL phase and is queued in ``self.waiting_etg``.
        Queued events are released upon entering INTERNAL phase and then processed.

        The actual processing is handled by the multiplexing scheme.

        If a SwapUpdate was received before processing this event and buffered in ``self.swap.waiting_su``,
        it is re-processed at this time.

        Args:
            event: Event containing the entangled qubit and its associated metadata (e.g., neighbor).

        """
        event.cancel()
        if self.node.timing.is_internal():
            self.qubit_is_entangled_in_internal(event)
        else:
            self.qubit_is_entangled_in_external(event)

    def qubit_is_entangled_in_external(self, event: QubitEntangledEvent) -> None:
        """
        Handle a qubit entering ENTANGLED state when in non-INTERNAL phase of SYNC timing mode.
        """
        self.entangled_in_external.append(event)

    def qubit_is_entangled_in_internal(self, event: QubitEntangledEvent) -> None:
        """
        Handle a qubit entering ENTANGLED state when in INTERNAL phase of SYNC timing mode or ASYNC timing mode.
        """
        self.cnt.n_entg += 1

        mq = event.qubit
        assert mq.state is QubitState.ENTANGLED1, f"unexpected state {mq.state}"
        assert mq.key
        mq.partner = event.neighbor, mq.key

        _, epr = self.memory.read(mq.addr, has=self.epr_type)
        assert not epr.orig_eprs, f"{mq} is not elementary entanglement"

        fp = self.qubit_is_entangled_next(mq, epr)
        self.log_debug("ENTANGLED %s path_id=%s | %s", mq, fp.path_id if fp else None, epr)

        if not mq.epr_path_ids:
            self.release_qubit(mq, need_remove=True)
            return

        match mq.state:
            case QubitState.PURIF:
                self.qubit_is_purif(mq, unwrap(fp), event.neighbor)
            case QubitState.ELIGIBLE:
                self.qubit_is_eligible(mq, fp)
            case _:
                # MuxSchemeStatistical may leave the qubit in ENTANGLED1 state
                assert mq.state is QubitState.ENTANGLED1, f"unexpected state {mq.state}"

        if mq.state is not QubitState.RELEASE:
            self.swap.pop_waiting_su(mq)

    @abstractmethod
    def qubit_is_entangled_next(self, mq: MemoryQubit, epr: Entanglement) -> FibPath | None:
        """
        Part of ``qubit_is_entangled_in_internal`` logic, overridden by subclass.

        If the entanglement is unwanted, keep ``mq.epr_path_ids`` unset or empty.
        The qubit would then be released.

        Otherwise, populate ``mq.epr_path_ids`` with potential paths, then either:

        * Change qubit state to PURIF, return a FIB path entry to guide purification.
        * Change qubit state to ELIGIBLE, to start swapping.
          If a FIB entry is returned, it is supplied to swap candidate selection function.
        """

    def qubit_is_purif(self, qubit: MemoryQubit, fp: FibPath, partner: QNode):
        """
        Handle a qubit entering PURIF state or have completed a previous purification round.

        1. Determines the segment in which the qubit is entangled and number of required purification rounds.
        2. If the required rounds are completed, the qubit becomes eligible.
        3. Otherwise, check if own node is primary for the purification protocol.
           If so, search for an auxiliary qubit to use, release the auxiliary qubit,
           and send PURIF_SOLICIT to the partner node.

        Args:
            qubit: The memory qubit at PURIF state.
            fp: FIB path entry containing routing and purification instructions.
            partner: The node with which the qubit shares an EPR.
        """
        assert qubit.state is QubitState.PURIF, f"unexpected state {qubit.state}"

        own_idx, own_rank = fp.own_idx, fp.own_swap_rank
        partner_idx, partner_rank = fp.find_index_and_swap_rank(partner.name)
        if own_rank > partner_rank:
            # swapping order disallows initiating purif / swap / consumption
            return

        segment_name = f"{self.node.name}-{partner.name}" if own_idx < partner_idx else f"{partner.name}-{self.node.name}"
        want_rounds = fp.purif.get(segment_name, 0)
        self.log_debug(
            "segment %s (qubit %s) has %s and needs %s purif rounds", segment_name, qubit.addr, qubit.purif_rounds, want_rounds
        )

        if qubit.purif_rounds == want_rounds:
            qubit.state = QubitState.ELIGIBLE
            self.qubit_is_eligible(qubit, fp)
            return
        assert qubit.purif_rounds < want_rounds

        is_primary = (own_rank, own_idx) < (partner_rank, partner_idx)
        if not is_primary:
            self.log_debug("is not primary node for segment %s purif", segment_name)
            return

        candidates = self.memory.find(
            lambda q, v: (
                q.addr != qubit.addr  # not the same qubit
                and q.state is QubitState.PURIF  # in PURIF state
                and q.purif_rounds == qubit.purif_rounds  # with same number of purif rounds
                and partner in (v.src, v.dst)  # with the same partner
                and q.path_id == fp.path_id  # on the same path_id
            ),
            has=self.epr_type,
        )
        found = call_select(candidates, self._select_purif_qubit, qubit, fp, partner)
        if not found:
            self.log_debug("no candidate EPR for segment %s purif round %s", segment_name, 1 + qubit.purif_rounds)
            return

        self.purif.start(qubit, found[0], fp, partner)

    def qubit_is_eligible(self, mq0: MemoryQubit, fp: FibPath | None):
        """
        Handle a qubit entering ELIGIBLE state.

        If this is an end node of the path, consume the EPR.

        Otherwise, update the EPR age cut-off scheme, and then attempt entanglement swapping:

        1. Look for a matching eligible qubit to perform swapping.
        2. Generate a new EPR if successful.
        3. Notify adjacent nodes with SWAP_UPDATE messages.

        Args:
            qubit: The qubit that became eligible.
            fp: FIB path entry (not available with MuxSchemeStatistical).
        """
        assert mq0.state is QubitState.ELIGIBLE, f"unexpected state {mq0.state}"
        self.cnt.n_eligible += 1
        mq0.purif_rounds = 0
        mq0.eligible_time = self.simulator.tc

        if not self.node.timing.is_internal():
            self.log_debug("INT phase is over -> stop swaps")
            return

        _, epr0 = self.memory.read(mq0.addr, has=self.epr_type)
        if self._try_consume(mq0, epr0, fp):
            return

        swap_decision = self.find_swap_with(mq0, epr0, fp)
        if swap_decision:
            mq1, fp = swap_decision
            self.cutoff.before_swap(mq0, mq1, fp)
            if fp.route.index(unwrap_cast(mq0.partner)[0].name) < fp.route.index(unwrap_cast(mq1.partner)[0].name):
                self.swap.start(mq0, mq1, fp)
            else:
                self.swap.start(mq1, mq0, fp)
        else:
            self.cutoff.before_store_eligible(mq0, PathDirection.L if epr0.dst is self.node else PathDirection.R, fp)

    def _try_consume(self, qubit: MemoryQubit, epr: Entanglement, fp: FibPath | None) -> bool:
        """
        If the EPR matches an end-to-end request, inform ``Consumer`` to consume the EPR.
        """
        now = self.simulator.tc
        if fp is None:
            # This branch is taken only when using MuxSchemeStatistical.
            # It searches for active requests spanning src,dst in either direction.
            # If none is found, the EPR cannot be consumed.
            frs = self.fib.list_end_reqs(unwrap_cast(qubit.partner)[0].name)
            if (fr := next((fr for fr in frs if fr.is_active(now)), None)) is None:
                return False
        elif not fp.own_is_end_node:
            # When own node is not an end node, the EPR should continue swapping and not be consumed.
            return False
        elif not (fr := fp.req).is_active(now):
            self.log_debug(
                "CONSUME_SKIP addr=%s key=%s partner=%s epr-count-remain=%s reason=req-inactive",
                qubit.addr,
                qubit.key,
                unwrap_cast(qubit.partner)[0].name,
                fr.epr_count_remain,
            )
            # If the FIB request entry is no longer active, consumption is disallowed.
            # Returning False would cause .qubit_is_eligible() to start swapping but own node is an end-node
            # for the FIB path entry so there's nothing to swap with.
            # Instead, we release the qubit and return True to prevent swapping.
            self.release_qubit(qubit, need_remove=True)
            return True

        # If epr_count is unrestricted, fr.epr_count_remain initializes as infinity and remains infinity.
        fr.epr_count_remain -= 1

        self.log_debug(
            "CONSUME_PASS addr=%s key=%s partner=%s epr-count-remain=%s",
            qubit.addr,
            qubit.key,
            qubit.partner,
            fr.epr_count_remain,
        )

        if fr.epr_count_remain == 0:
            self.request_reached_epr_count(fr)

        qubit.state = QubitState.CONSUME
        self.simulator.sched(QubitConsumeEvent(self.node, qubit, epr, t=self.simulator.tc, req_id=fr.req_id))
        return True

    @abstractmethod
    def find_swap_with(self, mq0: MemoryQubit, epr0: Entanglement, fp: FibPath | None) -> tuple[MemoryQubit, FibPath] | None:
        """
        Choose another qubit to swap with a qubit entering ELIGIBLE state and ready to swap.

        Args:
            mq0: The qubit entering ELIGIBLE state.
            epr0: The entanglement stored in the qubit.
            fp: FIB path entry found by ``qubit_is_entangled_next`` or used in last round of purification.

        Returns:
            None: Do not swap.
            [0]: The other qubit, which must be in ELIGIBLE state.
            [1]: FIB entry to guide ``ForwarderSwapProc``.
        """

    def request_reached_epr_count(self, fr: FibRequest) -> None:
        """
        Invoked on an end node when a request with ``epr_count`` restriction has reached this limit.
        """
        _ = fr

    @event_handler
    def qubit_is_decohered(self, event: MemoryDecohereEvent) -> None:
        event.cancel()
        assert self.node.timing.is_async(), f"unexpected {event} in SYNC timing mode, (t_ext+t_int) too high"
        self.release_qubit(event.qubit, is_decoh=True)

    def release_qubit(self, qubit: MemoryQubit, *, need_remove=False, is_decoh=False):
        """
        Release a qubit.

        Args:
            need_remove: Whether to remove the data associated with the qubit.
                         This should be set to True unless .read(remove=True) is already performed.
            is_decoh: Whether the release was caused by MemoryDecohereEvent.
        """
        if need_remove:
            self.memory.read(qubit.addr, remove=True)

        if is_decoh:
            self.swap.handle_decohere(unwrap_cast(qubit.key))

        qubit.state = QubitState.RELEASE
        event = QubitReleasedEvent(self.node, qubit, is_decoh=is_decoh, t=self.simulator.tc)
        # Set higher priority to prevent duplicate releases from decohere/cut-off and swap failure.
        event.priority = -1000
        self.simulator.sched(event)
