import math
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Final, Literal, final, override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.qchannel import QuantumChannel
from mqns.models.delay import ConstantDelayModel
from mqns.models.epr import Entanglement
from mqns.network.ll.counter import LinkLayerCounters
from mqns.network.ll.entangle import LLEntangleEvent, LLEntangleProc, LLPath, LLPathID, ReserveMsg, ReserveParsedMsg
from mqns.network.ll.entangle_2nd import LLEntangle2ndEvent
from mqns.network.ll.event import QubitReleasedEvent
from mqns.simulator import EventHandleSlot, Time, func_to_event
from mqns.utils import AutoIncrementIdentifier, unwrap, unwrap_cast

if TYPE_CHECKING:
    from mqns.network.ll.linklayer import LinkLayer

_AUTOID = AutoIncrementIdentifier("llk_")
"""
Automatically assigned ``RESERVE_REQ["key"]`` name.
"""


class LLPathPri(LLPath):
    oreq_table: Final[MutableMapping[str, "_EntangleTask"]]
    """
    Table of outgoing RESERVE_REQ sent by this node for which the EPR has not arrived.
    Key is reservation key.
    Value is EntangleTask record.
    """

    def __init__(self, path_id: LLPathID):
        super().__init__(path_id)
        self.oreq_table = {}

    @override
    def assert_is_empty(self) -> None:
        assert len(self.oreq_table) == 0, f"{self}.oreq_table = {self.oreq_table}"


@final
class LLEntangleProcPri(LLEntangleProc[LLPathPri]):
    """
    Entanglement generation procedure of one quantum channel on the primary node.
    """

    cnt: Final[LinkLayerCounters]

    def __init__(self, ll: "LinkLayer", ch: QuantumChannel):
        super().__init__(ll, ch)
        self.cnt = LinkLayerCounters(
            history=ll._cnt_history,
            rtt=ll.simulator.time(sec=ConstantDelayModel.extract(ch.delay) * 2),
        )

        self.min_attempt_spacing: Final[Time | None] = (
            None if math.isinf(ll.max_attempt_rate) else self.simulator.time(sec=1 / ll.max_attempt_rate)
        )
        """Minimum spacing between adjacent attempts on this channel, None means unlimited."""

        self.attempting: Final[dict[_EntangleTask, None]] = {}
        """
        Accepted reservations whose successful attempt has not started, used only under attempt rate throttling.
        This is an insertion-ordered set.
        """

        la = ch.link_arch

        # link_arch.set() may be called multiple times in several situations:
        # - qchannel is activated, deactivated, and re-activated.
        # - qchannel is activated from both sides due to shared paths in opposite directions.
        # Nevertheless, we assume every LinkLayer instance has the same parameters,
        # so that the parameters saved into LinkArch are the same every time.
        la.set(
            time_accuracy=self.simulator.accuracy,
            ch=ch,
            reset_time=self.ll.reset_time,
            tau_0=self.ll.tau_0,
            epr_type=self.node.network.epr_type,
            store_decays=(self.memory.time_decay, self.peer.memory.time_decay),
        )

        epr_tpl = la.make_epr(self.simulator.ts, self.node, self.peer, key=None)

        self.log_debug(
            "CHANNEL_ACTIVATE link_arch=%s attempt_interval=%s min_attempt_spacing=%s t_notify=%s,%s | template %s",
            la.name,
            la.attempt_interval,
            self.min_attempt_spacing,
            la.d_notify_pri,
            la.d_notify_2nd,
            epr_tpl,
        )

    @override
    def path_activate(self, path_id: LLPathID) -> None:
        super().path_activate(path_id)

        # Start reservations for this path_id if the network is in ASYNC timing mode.
        if self.node.timing.is_async():
            self.run_channel(path_id)

    @override
    def _path_activate_create(self, path_id: LLPathID) -> LLPathPri:
        return LLPathPri(path_id)

    @override
    def _path_deactivate_start_draining(self, lp: LLPathPri) -> Time:
        now = self.simulator.tc

        # No more attempts may start.
        # If the successful attempt has started or is starting in the same time slot, the task is allowed to continue.
        # Otherwise, the task is aborted and the secondary node is informed to release the reserved qubit.
        tasks_aborted: list[str] = []
        tasks_uncancelable: list[str] = []
        for key, task in lp.oreq_table.items():
            if task.k == -1:  # reservation has not completed
                tasks_aborted.append(key)
                continue
            if task.t_success <= now:  # successful attempt has started or is starting in the same time slot
                tasks_uncancelable.append(key)
                continue
            tasks_aborted.append(key)
            task.success_event.cancel()
            if task.notify_pri_event is not None:
                task.notify_pri_event.cancel()
            if task.notify_2nd_event is not None:
                task.notify_2nd_event.cancel()

        throttle_changed = False
        for key in tasks_aborted:
            task = lp.oreq_table.pop(key)
            task.mq.state = QubitState.RAW
            self.send_msg(ReserveMsg(cmd="RESERVE_ABORT", path_id=lp.path_id, key=key))
            if task in self.attempting:
                del self.attempting[task]
                throttle_changed = True

        # Remaining reservations on other paths may attempt more frequently.
        if throttle_changed:
            self._throttle_rebalance()

        self.log_debug("%s PATH_DRAIN abort-tasks=%s uncancelable-tasks=%s", lp, tasks_aborted, tasks_uncancelable)
        return now + self.memory.t_cohere

    @override
    def sync_external_enter(self) -> None:
        # Start reservations.
        self.run_channel()

    @override
    def sync_external_exit(self) -> None:
        # Clear incomplete reservations.
        for lp in self.paths_active.values():
            lp.oreq_table.clear()
        for task in self.attempting:
            task.success_event.cancel()
        self.attempting.clear()

    def run_channel(self, only_path_id: LLPathID | Literal[False] = False) -> None:
        """
        Start entanglement generation on qubits assigned to a channel.

        Args:
            only_path_id: If not False, only consider qubits allocated to a specified path.
        """
        qubits = self.memory.find(
            (lambda q, _: q.state is QubitState.RAW and q.path_id in self.paths_active)
            if only_path_id is False
            else (lambda q, _: q.state is QubitState.RAW and q.path_id == only_path_id),
            qchannel=self.ch,
        )
        for q, v in qubits:
            assert q.key is None
            assert v is None
            self.start_reservation(None, q)

    def start_reservation(self, lp: LLPathPri | None, mq: MemoryQubit) -> None:
        """
        Start entanglement generation on a qubit assigned to a channel.

        Args:
            lp: LLPathPri record, if known.
            mq: The qubit, which must be in RAW state.
        """
        lp = lp or self.paths_active[mq.path_id]

        # Generate a unique reservation key to represent this entanglement generation protocol execution
        # and the entanglement, which is valid until the entanglement is released.
        mq.key = _AUTOID()
        mq.state = QubitState.ACTIVE

        # Remember the pending reservation in LLPathPri record.
        lp.oreq_table[mq.key] = _EntangleTask(lp, mq)
        self.log_debug("RESERVE_REQ(%s, %s) sent mq-addr=%s", lp.path_id, mq.key, mq.addr)

        # Transmit RESERVE_REQ message to the secondary node.
        self.send_msg(ReserveMsg(cmd="RESERVE_REQ", path_id=mq.path_id, key=mq.key))

    def handle_reserve_res(self, req: ReserveParsedMsg) -> None:
        # Locate the LLPathPri record and the EntangleTask context.
        lp = self.paths_active.get(req.path_id)
        if lp is None:
            self.log_debug("%s ignored reason=no-active-path", req)
            return None
        task = lp.oreq_table.get(req.key, None)
        if task is None:
            self.log_debug("%s ignored reason=no-active-task", req)
            return
        assert task.k == -1, f"{task} reservation already accepted"

        # Mark the qubit as reserved.
        mq = task.mq
        assert mq.key == req.key
        mq.state = QubitState.RESERVED

        # Calculate which attempt would succeed.
        task.k = self.ll._calc_attempt(self.ch)
        task.t_reserve = self.simulator.tc

        if self.min_attempt_spacing is None:
            # Without throttling, attempts are spaced by link_arch attempt_interval.
            task.t_success = task.t_reserve + self.ch.link_arch.attempt_interval * (task.k - 1)
            self._prepare_epr(task)
            return

        # With throttling, the successful attempt time depends on other concurrent reservations.
        # It is recomputed whenever the set of attempting reservations changes.
        self.attempting[task] = None
        self._throttle_rebalance()

    def _prepare_epr(self, task: "_EntangleTask") -> None:
        """
        Schedule EPR delivery events, after ``task.t_success`` has been determined.
        """
        mq = task.mq
        la = self.ch.link_arch
        t_notify_pri = task.t_success + la.d_notify_pri
        t_notify_2nd = task.t_success + la.d_notify_2nd
        task.t_finish = max(t_notify_pri, t_notify_2nd)

        # When using SYNC timing, if the successful attempt cannot finish within the current EXTERNAL phase,
        # the EPR is not scheduled because it cannot arrive in time.
        # Memory reset and table cleanups are performed by EXTERNAL phase exit trigger.
        if not self.node.timing.is_external(task.t_finish):
            self.log_debug(
                "%s EPR_SKIP mq-addr=%s t_success=%s t_notify=%s,%s reason=beyond-external-phase",
                task,
                mq.addr,
                task.t_success,
                t_notify_pri,
                t_notify_2nd,
            )
            return

        # Schedule EPR delivery events on both nodes.
        epr = la.make_epr(task.t_success, self.node, self.peer, key=task.key)
        task.notify_pri_event = LLEntanglePriEvent(self, task, epr, t=t_notify_pri)
        task.notify_2nd_event = LLEntangle2ndEvent(task.key, epr, t=t_notify_2nd)
        self.simulator.sched(task.notify_pri_event)
        self.simulator.sched(task.notify_2nd_event)
        self.log_debug(
            "%s EPR_PREPARE mq-addr=%s epr=%s t_success=%s t_notify=%s,%s",
            task,
            mq.addr,
            epr.name,
            task.t_success,
            t_notify_pri,
            t_notify_2nd,
        )

    def _throttle_rebalance(self) -> None:
        """
        Recompute successful attempt time of every attempting reservation, under attempt rate throttling.

        If n reservations are making attempts concurrently, each reservation's attempts are spaced out by
        ``max(attempt_interval, n * min_attempt_spacing)``, so that the aggregate attempt rate on the channel
        does not exceed ``max_attempt_rate``.
        Whenever n changes, each reservation's progress (attempts made so far) is accumulated under the old interval,
        and its remaining attempts are rescheduled under the new interval.
        """
        assert self.min_attempt_spacing is not None
        now = self.simulator.tc
        interval = max(self.ch.link_arch.attempt_interval, self.min_attempt_spacing * len(self.attempting))

        for task in self.attempting:
            if task.interval.slot > 0:
                task.progress += (now.slot - task.t_progress.slot) / task.interval.slot
            elif task.interval is not Time.SENTINEL:
                task.progress = task.k - 1
            task.t_progress = now
            task.interval = interval

            remaining = max(0.0, task.k - 1 - task.progress)
            t_success = now + Time(round(remaining * interval.slot), accuracy=now.accuracy)
            if t_success == task.t_success:
                continue
            task.t_success = t_success
            self.simulator.sched(event := func_to_event(t_success, self._throttle_success, task))
            task.success_event.set(event)

        self.log_debug("THROTTLE n-attempting=%d interval=%s", len(self.attempting), interval)

    def _throttle_success(self, task: "_EntangleTask") -> None:
        """
        Handle the start of the successful attempt under attempt rate throttling.
        """
        del self.attempting[task]
        self._throttle_rebalance()
        self._prepare_epr(task)

    def before_deliver(self, task: "_EntangleTask") -> None:
        """
        Record a task prior to EPR delivery.
        """
        assert task.k >= 1, f"{task} reservation not accepted"
        del task.lp.oreq_table[task.key]

        self.cnt.n_etg += 1
        self.cnt.n_attempts += task.k
        if self.cnt._history is not None:
            self.cnt._history.append(task.t_finish.slot - task.t_reserve.slot)

    @override
    def _release_reuse_qubit(self, event: QubitReleasedEvent, lp: LLPathPri) -> None:
        self.log_debug("%s processed", event)
        if event.is_decoh:
            self.cnt.n_decoh += 1
        if self.node.timing.is_async():
            self.start_reservation(lp, event.qubit)


@final
class LLEntanglePriEvent(LLEntangleEvent):
    """
    Event to notify the primary node about successful elementary entanglement.
    """

    def __init__(
        self,
        proc: LLEntangleProcPri,
        task: "_EntangleTask",
        epr: Entanglement,
        *,
        t: Time,
    ):
        super().__init__(t, unwrap_cast(epr.src), unwrap_cast(epr.dst), task.key, epr)
        self.proc = proc
        self.task = task


class _EntangleTask:
    lp: Final[LLPathPri]
    mq: Final[MemoryQubit]
    k: int = -1
    """
    Which attempt shall success, starting with 1.
    ``-1`` means the reservation has not been accepted.
    """
    t_reserve: Time = Time.SENTINEL
    """
    Time point when the primary node receives the RESERVE_RES message and the first attempt begins.
    ``Time.SENTINEL`` means the reservation has not been accepted.
    """
    t_success: Time = Time.SENTINEL
    """
    Time point when the successful attempt begins.
    ``Time.SENTINEL`` means the reservation has not been accepted.
    """
    t_finish: Time = Time.SENTINEL
    """
    Time point when the last notify event occurs.
    ``Time.SENTINEL`` means the reservation has not been accepted.
    """
    notify_pri_event: LLEntanglePriEvent | None = None
    """
    Event to notify primary, cancelable before ``t_success``.
    """
    notify_2nd_event: LLEntangle2ndEvent | None = None
    """
    Event to notify secondary, cancelable before ``t_success``.
    """
    progress: float = 0.0
    """
    Number of attempts made as of ``t_progress``, used only under attempt rate throttling.
    """
    t_progress: Time = Time.SENTINEL
    """
    Time point when ``progress`` was last updated, used only under attempt rate throttling.
    """
    interval: Time = Time.SENTINEL
    """
    Current interval between adjacent attempts, used only under attempt rate throttling.
    """

    success_event: Final[EventHandleSlot]
    """
    Event at ``t_success``, used only under attempt rate throttling, cancelable before ``t_success``.
    """

    def __init__(self, lp: LLPathPri, mq: MemoryQubit):
        self.lp = lp
        self.mq = mq
        self.success_event = EventHandleSlot()

    @property
    def key(self) -> str:
        return unwrap(self.mq.key)

    def __repr__(self) -> str:
        return f"EntangleTask({self.key}, k={self.k}, t_reserve={self.t_reserve})"
