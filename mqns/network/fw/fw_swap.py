from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar, NamedTuple, override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.models.core import QuantumModel
from mqns.models.delay import DelayModel
from mqns.models.epr import Entanglement
from mqns.models.error import ErrorModel
from mqns.network.fw.fib import FibPath, FibSwapGroup
from mqns.network.fw.fw_module import ForwarderModule, fw_signaling_cmd_handler
from mqns.network.fw.message import SwapUpdateMsg
from mqns.simulator import Event, Time, func_to_event
from mqns.utils import unwrap, unwrap_cast

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class _SwapArm(NamedTuple):
    mq: MemoryQubit
    """Local qubit entangled with partner."""
    o_key: str
    """Local qubit reservation key."""
    p_key: str
    """Partner qubit reservation key."""

    @staticmethod
    def new(mq: MemoryQubit) -> "_SwapArm":
        o_key = unwrap(mq.key)
        p_key = o_key if mq.partner is None else mq.partner.key
        return _SwapArm(mq, o_key, p_key)

    def __repr__(self) -> str:
        return f"SwapArm(qubit={self.mq.addr}, o-key={self.o_key}, p-key={self.p_key})"


class _SwapHerald(NamedTuple):
    """Heralding instruction."""

    su: SwapUpdateMsg
    """
    Signaling message.
    The ``.path_id`` field has a dummy value and must be replaced.
    Modifying this message directly without copying is permissible.
    """
    dest: str
    """Destination node name."""
    paths: Sequence[int]
    """Possible path IDs for classical signaling, must be non-empty."""


class _SwapParsedUpdate(NamedTuple):
    """SWAP_UPDATE parsed in own node context."""

    fp: FibPath
    expiry: int
    q_paths: Sequence[int]

    swapper: str
    swapper_rank: int

    o_key: str
    partner: str
    p_key: str


class _SwapTask:
    """
    Track the logical progress of a swap process within a swap group.

    An instance of ``SwapTask`` is lazily created when a node starts swapping two EPRs (either elementary EPR or
    completed EPR from lower-ranked swap group). The instance tracks how many nodes in the same ``FibSwapGroup``
    have completed their swaps, based on local operation outcomes and heralded information learned from other nodes
    within the swap group.
    """

    UNKNOWN: ClassVar = "#UNK"
    """Indicate an unknown value, only used in failure heralding."""

    proc: "ForwarderSwapProc"
    """``ForwarderSwapProc`` reference."""
    sg: FibSwapGroup
    """Swap group from primary FIB entry."""

    o_started = False
    """Is my own local swap started?"""
    o_complete = False
    """Is my own local swap completed?"""
    l_complete = False
    """Is the chain to my left completed?"""
    r_complete = False
    """Is the chain to my right completed?"""

    expiry: int | None = None
    """Time slot for group-wise EPR expiration, zero on swap failure anywhere."""
    lc_paths: list[int] | None = None
    """Possible path IDs for leftward classical signaling."""
    rc_paths: list[int] | None = None
    """Possible path IDs for rightward classical signaling."""
    q_paths: list[int] | None = None
    """Possible path IDs for the entanglement (sorted)."""

    l_sent = False
    """Is the left same-rank peer or higher-rank neighbor heralded?"""
    r_sent = False
    """Is the right same-rank peer or higher-rank neighbor heralded?"""

    la_key = UNKNOWN
    """Qubit reservation key between own node and ``sg.l_adj``."""
    ra_key = UNKNOWN
    """Qubit reservation key between own node and ``sg.r_adj``."""

    lb_key = UNKNOWN
    """Qubit reservation key between ``sg.nodes[0]`` and ``sg.l_neigh``."""
    rb_key = UNKNOWN
    """Qubit reservation key between ``sg.nodes[-1]`` and ``sg.r_neigh``."""

    has_expire_event = False
    """Whether ``ForwarderSwapProc._u_expire_task`` is scheduled."""

    def __init__(self, proc: "ForwarderSwapProc", fp: FibPath):
        self.proc = proc
        self.sg = unwrap(fp.sg)

    def begin_local_swap(self, la: _SwapArm, ra: _SwapArm) -> None:
        """
        Record start of local swap.

        Args:
            la: Left swap arm.
            ra: Right swap arm.
        """

        self.o_started = True
        self.lc_paths = unwrap_cast(la.mq.epr_path_ids)
        self.rc_paths = unwrap_cast(ra.mq.epr_path_ids)
        self._update_q_paths(self.lc_paths)
        self._update_q_paths(self.rc_paths)

        # Choose a SwapGroup based on chosen swap arms.
        # If own node is leftmost/rightmost in the SwapGroup, we won't hear from left/right peer.
        self._pivot_sg()
        if self.sg.l_most:
            self.l_complete = True
        if self.sg.r_most:
            self.r_complete = True

        # These EPRs are known by both own node and the adjacent nodes.
        self.la_key = la.p_key
        self.ra_key = ra.p_key

        # If this node is the leftmost or rightmost with a group, these EPRs are known by the higher-ranked neighbors.
        if self.sg.l_most:
            self.lb_key = self.la_key
        if self.sg.r_most:
            self.rb_key = self.ra_key

    def end_local_swap(self, expiry: int) -> list[_SwapHerald]:
        """
        Save local swap outcome.

        Args:
            expiry: Zero on swap failure, otherwise time slot for swapped EPR expiration time.

        Returns:
            Heralding instructions, if any.
        """
        assert self.o_started, "{self}: end_local_swap without start_local_swap"

        self._update_expiry(expiry)

        self.o_complete = True

        # If local swap failed, there's no need to wait for heralding.
        if expiry == 0:
            self.l_complete = True
            self.r_complete = True

        return self._check_triggers()

    def notify_remote_swap(self, su: _SwapParsedUpdate) -> list[_SwapHerald]:
        """
        Save heralded swap outcome.

        Args:
            su: SWAP_UPDATE message.

        Returns:
            Heralding instructions, if any.
        """
        self._update_expiry(su.expiry)
        self._update_q_paths(su.q_paths)

        sg = self.sg
        if su.swapper == sg.l_adj:  # Message came from left peer.
            self.l_complete = True
            self.lb_key = su.p_key
            self.la_key = su.o_key
            if su.expiry == 0:
                # Left peer does not need to hear about the failure they just told us.
                self.l_sent = True
        elif su.swapper == sg.r_adj:  # Message came from right peer.
            self.r_complete = True
            self.rb_key = su.p_key
            self.ra_key = su.o_key
            if su.expiry == 0:
                # Right peer does not need to hear about the failure they just told us.
                self.r_sent = True
        else:
            raise RuntimeError(f"SwapGroup({sg.nodes},{sg.own_idx}) received swap outcome from unexpected node {su.swapper}")

        self._pivot_sg()
        return self._check_triggers()

    def _update_expiry(self, expiry: int) -> None:
        if self.expiry is None:
            self.expiry = expiry
        else:
            self.expiry = min(self.expiry, expiry)

    def _update_q_paths(self, input: Iterable[int]) -> None:
        if self.q_paths is None:
            self.q_paths = sorted(input)
        else:
            # Intersect q_paths with the input.
            self.q_paths = [i for i in self.q_paths if i in input]

    def _pivot_sg(self) -> None:
        """
        If current SwapGroup is no longer a viable path but there are other viable paths,
        pivot SwapGroup to another viable path.
        This mechanism can only trigger when used with ``MuxSchemeStatistical``.
        """
        if not self.q_paths or self.sg.path_id in self.q_paths:
            return

        fp = self.proc.fib.get_path(self.q_paths[0])
        sg = unwrap(fp.sg)
        if sg.l_adj == self.sg.r_adj or sg.r_adj == self.sg.l_adj:
            self.l_complete, self.r_complete = self.r_complete, self.l_complete
            self.lc_paths, self.rc_paths = self.rc_paths, self.lc_paths
            self.l_sent, self.r_sent = self.r_sent, self.l_sent
            self.la_key, self.ra_key = self.ra_key, self.la_key
            self.lb_key, self.rb_key = self.rb_key, self.lb_key
        self.sg = sg

    def _check_triggers(self) -> list[_SwapHerald]:
        sg = self.sg
        heralds: list[_SwapHerald] = []
        spoiled = self.expiry == 0 or not self.q_paths

        if (
            not self.l_sent  # leftward herald is allowed at most once
            and (
                self.o_started  # in case of swap failure, leftward heralding requires la_key+lb_key
                if self.expiry == 0
                # in the absence of swap failure:
                else (
                    self.r_complete  # (1) if there's a right peer, we must know their swap outcome and rb_key
                    and self.o_complete  # (2) we must know the local swap outcome
                    and sg.dir in ("l", "b")  # (3) swap group logic requires heralding leftward
                )
            )
        ):
            su = self._make_su(
                [
                    sg.l_adj,
                    self.lb_key if sg.l_most else self.la_key,
                    self.UNKNOWN if spoiled else sg.r_neigh,
                    self.UNKNOWN if spoiled else self.rb_key,
                ]
            )
            heralds.append(_SwapHerald(su, sg.l_adj, unwrap_cast(self.lc_paths)))
            self.l_sent = True

        if not self.r_sent and (
            self.o_started if self.expiry == 0 else (self.l_complete and self.o_complete and sg.dir in ("r", "b"))
        ):
            su = self._make_su(
                [
                    sg.r_adj,
                    self.rb_key if sg.r_most else self.ra_key,
                    self.UNKNOWN if spoiled else sg.l_neigh,
                    self.UNKNOWN if spoiled else self.lb_key,
                ]
            )
            heralds.append(_SwapHerald(su, sg.r_adj, unwrap_cast(self.rc_paths)))
            self.r_sent = True

        return heralds

    def _make_su(self, ends: list[str]) -> SwapUpdateMsg:
        sg = self.sg
        return SwapUpdateMsg(
            cmd="SWAP_UPDATE",
            path_id=-1,
            swapper=sg.own_node,
            ends=ends,
            expiry=unwrap_cast(self.expiry),
            # .q_paths is set in .begin_local_swap(), which is a prerequisite of .o_started and .o_complete
            q_paths=unwrap_cast(self.q_paths),
        )

    def __repr__(self) -> str:
        return (
            f"SwapTask(path_id={self.sg.path_id}, group={self.sg.repr_route()}, "
            f"c-paths={self.lc_paths}:{self.rc_paths}, q-paths={self.q_paths}, "
            f"complete={self.l_complete and 'l' or '_'}"
            f"{self.o_complete and 'o' or (self.o_started and 'O' or '_')}"
            f"{self.r_complete and 'r' or '_'}, "
            f"sent={self.l_sent and 'l' or '_'}{self.r_sent and 'r' or '_'}, "
            f"qubit-key={self.lb_key},{self.la_key},{self.ra_key},{self.rb_key}, expiry={self.expiry})"
        )


class _SwapTaskExpireEvent(Event):
    def __init__(self, task: _SwapTask, *, t: Time):
        super().__init__(t)
        self.task = task

    @override
    def invoke(self) -> None:
        self.task.proc._expire_task(self)

    def __repr__(self) -> str:
        return f"SwapTaskExpireEvent({self.task})"


class ForwarderSwapProc(ForwarderModule):
    """
    Part of ``Forwarder`` logic related to swapping procedure.

    Known assumptions and limitations:

    * ClassicChannel cannot have packet loss for SWAP_UPDATE.
      Otherwise, internal data structures will have memory leaks.
    * Parallel swapping (ASAP) is only permitted at rank 0.
      ``[2,0,0,1,0,0,2]`` is allowed; ``[2,0,1,0,1,0,2]`` is disallowed due to parallel swapping at rank 1.
    """

    table_leak_tol: ClassVar[int] = -1
    """
    Tolerance of internal data structure memory leak at end of a finite simulation.

    If this is set to a non-negative number, trigger an assertion error if any internal data structure
    has more than this number of leftover entries at end of simulation.
    This has no effect if set to a negative number or the simulation is continuous.
    """

    def __init__(self, *, ps: float, delay: DelayModel, error: ErrorModel, error_at_finish: bool):
        self.ps = ps
        """Probability of successful entanglement swapping."""
        assert 0.0 <= self.ps <= 1.0
        self.delay = delay
        """Swapping delay model."""
        self.error = error
        """Swapping error model."""
        self.error_at_finish = error_at_finish
        """Whether to apply memory errors at swap finish time instead of swap start time."""

        self.waiting_su: dict[str, _SwapParsedUpdate] = {}
        """
        SwapUpdates received prior to QubitEntangledEvent.

        * Key: MemoryQubit reservation key.
        * Value: SWAP_UPDATE message and related fields.
        """

        self.remote_swapped: dict[str, Entanglement] = {}
        """
        EPRs that have been swapped remotely but the SwapUpdateMsg have not arrived.

        * Key: MemoryQubit reservation key, for which the node expects an incoming SWAP_UPDATE message.
        * Value: Physical EPR object.
        """

        self.task_by_qubit: dict[str, _SwapTask] = {}
        """
        SwapTask associated with a memory qubit.

        * Key: MemoryQubit reservation key.
        * Value: SwapTask instance.
        """

    def install(self, fw: "Forwarder") -> None:
        super().install(fw)

        if not self.simulator.is_continuous:
            event = func_to_event(self.simulator.te, self.check_table_leak)
            event.priority = 0x1FFFFFFF
            self.simulator.sched(event)

    def check_table_leak(self) -> None:
        """
        Check for memory leak in internal data structures with ``table_leak_tol`` tolerance.

        This function is scheduled automatically at the end of a finite simulation.
        """
        max_table_size = 0
        for attr in ("waiting_su", "remote_swapped", "task_by_qubit"):
            table = getattr(self, attr)
            if n := len(table):
                self.log_warning("%s is not empty (len=%s): %s", attr, n, table)
                max_table_size = max(max_table_size, n)

        if self.table_leak_tol >= 0 and max_table_size > self.table_leak_tol:
            raise MemoryError("memory leak detected in data structures")

    def sync_internal_exit(self) -> None:
        """
        In SYNC timing mode, exit INTERNAL phase.
        """
        # Clear states.
        # All memory qubits are being discarded by LinkLayer, so that these have become useless.
        assert not self.waiting_su, f"waiting_su is not empty {self.waiting_su}"
        self.remote_swapped.clear()
        self.task_by_qubit.clear()

    def handle_decohere(self, key: str) -> None:
        """
        Cleanup state when releasing a qubit due to decoherence or CutOffScheme.

        Args:
            key: Qubit key.
        """
        deleted_from: list[str] = []
        if epr := self.remote_swapped.pop(key, None):
            epr.is_decohered = True
            deleted_from.append(f"remote_swapped[{key}]")
        if self.task_by_qubit.pop(key, None):
            deleted_from.append(f"task_by_qubit[{key}]")
        self.log_debug("DECOHERE key=%s deleted-from=%s", key, deleted_from)

    def _heralds(self, heralds: list[_SwapHerald]) -> None:
        """
        Send heralding messages.
        """
        for h in heralds:
            fp = None
            for p in h.paths:
                fib_path = self.fib.get_path(p)
                if h.dest in fib_path.route:
                    fp = fib_path
                    break

            if not fp:
                raise RuntimeError(f"{self}: cannot herald {h.dest} among paths {h.paths} | {h.su}")

            h.su["path_id"] = fp.path_id
            self.send_msg(self.network.get_node(h.dest), h.su, fp)

    def start(self, mq0: MemoryQubit, mq1: MemoryQubit, fp: FibPath) -> None:
        """
        Start swapping between two memory qubits.

        Args:
            mq0: Left qubit.
            mq1: Right qubit.
            fp: FIB path entry.
        """
        assert mq0.addr != mq1.addr
        assert unwrap_cast(mq0.partner).ch_peer != unwrap_cast(mq1.partner).ch_peer

        # Retrieve both qubits.
        arms = [_SwapArm.new(mq0), _SwapArm.new(mq1)]

        # Record local swap start in SwapTask.
        task, task_from = self._s_get_task(fp, arms)
        task.begin_local_swap(*arms)
        task_saved = self._s_put_task(task, arms)

        # Set SWAPPING state, so that forwarder cannot start another swapping on the same qubit.
        # The ALLOWED_STATE_TRANSITIONS matrix verifies existing state is ELIGIBLE.
        mq0.state = QubitState.SWAPPING
        mq1.state = QubitState.SWAPPING

        # Schedule swap completion event.
        # If the FIB entry is being uninstalled, swap delay is bypassed and local swap is deemed failed.
        now = self.simulator.tc
        if fp.req.is_active(now):
            finish_time = now + self.delay.calculate()
        else:
            finish_time = now
        error_time = finish_time if self.error_at_finish else now
        self.simulator.sched(func_to_event(finish_time, self._s_finish, arms, fp, error_time, task))

        self.log_debug("SWAP_START %s retrieved-from=%s saved-at=%s finish-time=%s", task, task_from, task_saved, finish_time)

    def _s_get_task(
        self, fp: FibPath, arms: Sequence[_SwapArm], task_via_event: _SwapTask | None = None
    ) -> tuple[_SwapTask, str]:
        task: _SwapTask | None = None
        task_from: list[str] = []
        for arm in arms:
            if t := self.task_by_qubit.pop(arm.o_key, None):
                task = t
                task_from.append(f"task_by_qubit[{arm.o_key}]")
        if task:
            return task, ",".join(task_from)

        if task_via_event:
            return task_via_event, "event"
        return _SwapTask(self, fp), "constructor"

    def _s_put_task(self, task: _SwapTask, arms: Sequence[_SwapArm]) -> list[str]:
        task_saved: list[str] = []
        for i, dir_complete in enumerate((task.l_complete, task.r_complete)):
            if dir_complete:
                continue
            arm = arms[i]
            self.task_by_qubit[arm.o_key] = task
            task_saved.append(f"task_by_qubit[{arm.o_key}]")
        return task_saved

    def _s_finish(self, arms: Sequence[_SwapArm], fp: FibPath, error_t: Time, task_via_event: _SwapTask) -> None:
        """
        Complete swapping between two memory qubits.

        This is scheduled by ``.start()`` after ``swap_delay``.
        """

        # Retrieve physical EPRs.
        local_expiry: list[int] = []
        alive_arms: list[_SwapArm] = []
        phy_eprs: list[Entanglement] = []
        for arm in arms:
            if arm.mq.state is not QubitState.SWAPPING or arm.mq.key != arm.o_key:
                # Elementary EPR decohered and potentially replaced.
                continue
            alive_arms.append(arm)
            _, epr = self.memory.read(arm.mq.addr, has=self.epr_type, remove=True)
            local_expiry.append(epr.decohere_time.slot)
            phy = self.remote_swapped.pop(arm.o_key, epr)
            phy_eprs.append(phy)

        # If the FIB entry is being uninstalled, local swap is deemed failed.
        if not fp.req.is_active(self.simulator.tc):
            phy_eprs.clear()

        # Attempt physical swap.
        new_epr, outcome_str, local_success = self._s_physical_swap(error_t, phy_eprs)
        self.log_debug("%s rank=%s | %s x %s = %s", outcome_str, fp.own_swap_rank, arms[0], arms[1], new_epr)

        # Release consumed qubits.
        for arm in alive_arms:
            self.fw.release_qubit(arm.mq)

        # Record local swap outcome in SwapTask, and then send heralding if allowed.
        task, task_from = self._s_get_task(fp, arms, task_via_event)
        heralds = task.end_local_swap(min(local_expiry) if local_success else 0)
        task_saved = self._s_put_task(task, arms)
        if task_saved:
            self.sched_expire_task(task)

        self.log_debug("SWAP_FINISH %s retrieved-from=%s saved-at=%s", task, task_from, task_saved)
        self._heralds(heralds)

    def _s_physical_swap(self, error_t: Time, phy_eprs: Sequence[Entanglement]) -> tuple[Entanglement | None, str, bool]:
        # If either memory qubit has decohered, abort the swap.
        if len(phy_eprs) != 2:
            return None, "SWAP_ABORT", False

        epr0, epr1 = phy_eprs
        flip0 = 0
        flip1 = 0
        if epr0.dst is not self.node:
            flip0 = 1
            epr0.flip_direction()
        if epr1.src is not self.node:
            flip1 = 1
            epr1.flip_direction()

        # Attempt the physical swap.
        # Memory error model is applied as of the specified time point.
        new_epr, local_success = Entanglement.swap(epr0, epr1, now=error_t, ps=self.ps, error=self.error)

        # Update physical swap counters.
        if local_success:
            self.fw_cnt.n_swapped += 1
        else:
            self.fw_cnt.n_swap_fail += 1

        # Deposit physical swap result.
        self._s_physical_deposit(new_epr)

        return new_epr, f"SWAP_{'SUCC' if local_success else 'FAIL'}_{flip0}{flip1}", local_success

    def _s_physical_deposit(self, new_epr: Entanglement) -> None:
        if new_epr.is_decohered:
            self.log_debug("physical deposit skipped reason=DECOHERED")
            return

        deposit_at: list[str] = []
        assert new_epr.orig_eprs
        for target_attr, key_i in (new_epr.src, 0), (new_epr.dst, 1):
            target = unwrap_cast(target_attr)
            key = new_epr.mem_keys[key_i]
            target.get_app(type(self.fw)).swap.remote_swapped[key] = new_epr
            deposit_at.append(f"{target.name}.remote_swapped[{key}]")
        self.log_debug("physical deposit at %s", ", ".join(deposit_at))

    def pop_waiting_su(self, qubit: MemoryQubit) -> None:
        """
        Invoked by ``Forwarder.qubit_is_entangled()`` after QubitEntangledEvent to process buffered SWAP_UPDATE.

        It's possible for SWAP_UPDATE to arrive in the same time slot as QubitEntangledEvent, for example in
        S-R-D linear topology, when node R performs a swap as soon as it is notified about R-D entanglement,
        before D is notified. This cannot be resolved with ``Event.priority`` mechanism because R and D may be
        notified at different times depending on the link architecture.
        """
        su = self.waiting_su.pop(unwrap(qubit.key), None)
        if (
            qubit.state is not QubitState.RELEASE  # qubit was released due to uninstalled path
            and su
        ):
            self.handle_update(su)

    @fw_signaling_cmd_handler("SWAP_UPDATE")
    def receive_update(self, msg: SwapUpdateMsg, fp: FibPath) -> None:
        """
        Receive an SWAP_UPDATE signaling message.
        """
        ends = msg["ends"]
        if ends[0] == self.node.name:
            _, o_key, partner, p_key = ends
        elif ends[2] == self.node.name:
            partner, p_key, _, o_key = ends
        else:
            raise ValueError(f"SWAP_UPDATE does not address node {self.node.name} | {msg}")

        swapper = msg["swapper"]
        q_paths = msg["q_paths"]
        if partner not in fp.route and q_paths:
            for path_id in q_paths:
                fe = self.fib.get_path(path_id)
                if swapper in fe.route and partner in fe.route:
                    fp = fe
                    break
        swapper_rank = fp.swap[fp.route.index(swapper)]

        self.handle_update(
            _SwapParsedUpdate(
                fp=fp,
                expiry=msg["expiry"],
                q_paths=q_paths,
                swapper=swapper,
                swapper_rank=swapper_rank,
                o_key=o_key,
                partner=partner,
                p_key=p_key,
            )
        )

    def handle_update(self, su: _SwapParsedUpdate) -> None:
        """
        Process a received or dequeued SWAP_UPDATE signaling message.
        """
        if not self.node.timing.is_internal():
            self.log_debug("INT phase is over -> stop swaps")
            return

        # Defer after QubitEntangledEvent arrives.
        if (qubit_pair := self.memory.read(su.o_key)) and qubit_pair[0].state is QubitState.RESERVED:
            self.waiting_su[su.o_key] = su
            return

        # If the swapper has a lower rank, it indicates the completion of a lower-ranked swap task.
        # Save its outcome into memory, so that this node can start purification and own-rank swapping,
        if su.swapper_rank < su.fp.own_swap_rank:
            self._u_lower(su, qubit_pair)
            return

        # If the swapper has the same rank, it is part of the swap task that is potentially parallel.
        # qubit_pair would be None, if own node has already swapped.
        assert su.swapper_rank == su.fp.own_swap_rank
        self._u_same(su, qubit_pair)

    def _u_lower(
        self,
        su: _SwapParsedUpdate,
        qubit_pair: tuple[MemoryQubit, QuantumModel | None] | None,
    ) -> None:
        """Process SWAP_UPDATE sent from a lower-ranked node."""

        # Retrieve qubit and new physical EPR.
        mq = qubit_pair[0] if qubit_pair else None
        new_phy = self.remote_swapped.pop(su.o_key, None)

        # If the lower-ranked swap failed, had conflict path, or the new EPR has decohered, release the qubit.
        if not su.q_paths or su.expiry <= self.simulator.tc.slot:
            if mq:
                if su.expiry == 0:
                    self.fw_cnt.n_su_lower[3] += 1
                    self.log_debug("releasing qubit %s reason=lower-swap-failure key=%s | %s", mq.addr, su.o_key, new_phy)
                elif not su.q_paths:
                    self.fw_cnt.n_su_lower[4] += 1
                    self.log_debug("releasing qubit %s reason=lower-swap-conflict key=%s | %s", mq.addr, su.o_key, new_phy)
                else:
                    self.fw_cnt.n_su_lower[2] += 1
                    self.log_debug(
                        "releasing qubit %s reason=lower-expiry expiry=%s key=%s | %s",
                        mq.addr,
                        self.simulator.time(slot=su.expiry),
                        su.o_key,
                        new_phy,
                    )
                self.fw.release_qubit(mq, need_remove=True)
            else:
                self.fw_cnt.n_su_lower[1] += 1
                self.log_debug("qubit decohered during SWAP_UPDATE transmission key=%s | %s", su.o_key, new_phy)
            return
        assert mq, f"qubit not found for {su.o_key}"
        assert new_phy, f"new_phy not found for {su.o_key}"

        # Verify that the new physical EPR matches the heralded EPR segment.
        # This logic only supports ASAP parallel swap at the lowest rank.
        # Having parallel swap at any higher rank would break the assertion, because the physical EPR
        # may have been swapped by a peer when own node processes SWAP_UPDATE.
        if new_phy.src is self.node:
            partner = unwrap_cast(new_phy.dst)
        elif new_phy.dst is self.node:
            partner = unwrap_cast(new_phy.src)
        else:
            raise ValueError(f"new_phy does not match node {self.node.name} | {new_phy}")
        assert partner.name == su.partner

        assert su.partner in su.fp.route

        # Store new EPR.
        mq.partner = unwrap_cast(mq.partner).with_far_side(partner, su.p_key)
        self.memory.write(mq.addr, new_phy, replace=True)

        self.fw_cnt.n_su_lower[0] += 1
        self.log_debug(
            "segment %s-%s swap completed for rank %s",
            unwrap_cast(new_phy.src).name,
            unwrap_cast(new_phy.dst).name,
            su.swapper_rank,
        )

        # Progress toward purification and this-rank swap.
        mq.purif_rounds = 0
        mq.state = QubitState.PURIF
        self.fw.qubit_is_purif(mq, su.fp, partner)

    def _u_same(
        self,
        su: _SwapParsedUpdate,
        qubit_pair: tuple[MemoryQubit, QuantumModel | None] | None,
    ) -> None:
        """Process SWAP_UPDATE sent from a same-ranked node."""

        # Retrieve SwapTask.
        task, task_from = self._u_get_task(su.fp, su.o_key)

        # If qubit does not exist, it means own node had swap failure previously and already notified both sides,
        # but the remote swap occurred while the outgoing SWAP_UPDATE is still in flight.
        if not task.o_complete and qubit_pair is None:
            self.fw_cnt.n_su_same[1] += 1
            deleted_from = None
            if self.remote_swapped.pop(su.o_key, None):
                deleted_from = f"remote_swapped[{su.o_key}]"
            self.log_debug(
                "SWAP_UPDATE_SAME %s retrieved-from=%s DROPPED reason=previous-swap-failure deleted-from=%s",
                task,
                task_from,
                deleted_from,
            )
            assert task_from == "constructor"
            return

        # Record heralded swap outcome in SwapTask, and then send heralding if allowed.
        heralds = task.notify_remote_swap(su)
        task_saved = None
        if not task.o_complete:
            self.task_by_qubit[su.o_key] = task
            task_saved = f"task_by_qubit[{su.o_key}]"
            self.sched_expire_task(task)

            # Expose the narrowed q_paths to qubit selection algorithm.
            mq, _ = self.memory.read(su.o_key, must=True)
            mq.epr_path_ids = task.q_paths

        self.fw_cnt.n_su_same[0] += 1
        self.log_debug("SWAP_UPDATE_SAME %s retrieved-from=%s saved-at=%s", task, task_from, task_saved)
        self._heralds(heralds)

    def _u_get_task(self, fp: FibPath, qubit_key: str) -> tuple[_SwapTask, str]:
        if t := self.task_by_qubit.pop(qubit_key, None):
            return t, f"task_by_qubit[qubit_key:={qubit_key}]"
        return _SwapTask(self, fp), "constructor"

    def sched_expire_task(self, task: _SwapTask) -> None:
        if task.has_expire_event:
            return
        task.has_expire_event = True
        t = self.simulator.tc if not task.expiry else self.simulator.time(slot=task.expiry)
        t += self.memory.t_cohere  # delay deletion so that incoming messages can be replied to
        self.simulator.sched(_SwapTaskExpireEvent(task, t=t))

    def _expire_task(self, event: _SwapTaskExpireEvent) -> None:
        task = event.task
        deleted_from: list[str] = []
        for key in task.la_key, task.ra_key:
            if key != _SwapTask.UNKNOWN and self.task_by_qubit.pop(key, None):
                deleted_from.append(key)
        if deleted_from:
            self.log_debug("TASK_EXPIRE %s deleted-from=%s", task, deleted_from)
