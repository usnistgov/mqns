from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Final, final, override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.models.delay import ConstantDelayModel
from mqns.models.epr import Entanglement
from mqns.network.ll.counter import LinkLayerCounters
from mqns.network.ll.entangle import LLEntangleProc, LLGroup, LLNotifyEvent, ReserveMsg
from mqns.network.ll.entangle_2nd import LLNotify2ndEvent
from mqns.network.ll.event import LLActivateEvent, OnAttemptSuccess, QubitReleasedEvent
from mqns.simulator import Event, Time
from mqns.utils import AutoIncrementIdentifier, unwrap, unwrap_cast

if TYPE_CHECKING:
    from mqns.network.ll.linklayer import LinkLayer

_AUTOID = AutoIncrementIdentifier("llk_")
"""
Automatically assigned ``RESERVE_REQ["key"]`` name.
"""


@final
class LLEntangleProcPri(LLEntangleProc["LLGroupPri"]):
    """
    Entanglement generation procedure of one quantum channel on the primary node.
    """

    cnt: Final[LinkLayerCounters]

    def __init__(self, ll: "LinkLayer", event: LLActivateEvent):
        super().__init__(ll, event, ch_repr_attrs=("node", "peer"))
        self.cnt = LinkLayerCounters(
            history=ll._cnt_history,
            rtt=ll.simulator.time(sec=ConstantDelayModel.extract(self.ch.delay) * 2),
        )

        la = self.ch.link_arch

        # link_arch.set() may be called multiple times in several situations:
        # - qchannel is activated, deactivated, and re-activated.
        # - qchannel is activated from both sides due to shared paths in opposite directions.
        # Nevertheless, we assume every LinkLayer instance has the same parameters,
        # so that the parameters saved into LinkArch are the same every time.
        la.set(
            time_accuracy=self.simulator.accuracy,
            ch=self.ch,
            reset_time=self.ll.reset_time,
            tau_0=self.ll.tau_0,
            epr_type=self.node.network.epr_type,
            store_decays=(self.node.memory.time_decay, self.peer.memory.time_decay),
        )

        epr_tpl = la.make_epr(self.simulator.ts, self.node, self.peer, key=None)

        self.log_debug(
            "CHANNEL_ACTIVATE link_arch=%s attempt_interval=%s t_notify=%s,%s | template %s",
            la.name,
            la.attempt_interval,
            la.d_notify_pri,
            la.d_notify_2nd,
            epr_tpl,
        )

    @override
    def _group_create(self, event: LLActivateEvent) -> "LLGroupPri":
        return LLGroupPri(self, event.gid, event.on_attempt_success)


class LLGroupPri(LLGroup[LLEntangleProcPri]):
    oreq_table: Final[MutableMapping[str, "_EntangleTask"]]
    """
    Table of outgoing RESERVE_REQ sent by this node for which the EPR has not arrived.
    Key is reservation key.
    Value is EntangleTask record.
    """

    on_attempt_success: Final[OnAttemptSuccess]
    """
    Multiplexing policy callback supplied by forwarding.
    Invoked upon starting each successful attempt.
    """

    def __init__(self, proc: LLEntangleProcPri, gid: int, on_attempt_success: OnAttemptSuccess):
        super().__init__(proc, gid)
        self.oreq_table = {}
        self.on_attempt_success = on_attempt_success

    @override
    def drain(self) -> Time:
        proc = self.proc
        now = proc.simulator.tc

        # No more attempts may start.
        # If the successful attempt has started or is starting in the same time slot, the task is allowed to continue.
        # Otherwise, the task is aborted and the secondary node is informed to release the reserved qubit.
        tasks_aborted: list[str] = []
        tasks_uncancelable: list[str] = []
        for key, task in self.oreq_table.items():
            success_event = task.success_event
            if success_event is None:  # reservation has not completed
                tasks_aborted.append(key)
            elif success_event.t <= now:  # successful attempt has started or is starting in the same time slot
                tasks_uncancelable.append(key)
            else:
                tasks_aborted.append(key)
                success_event.cancel()

        for key in tasks_aborted:
            task = self.oreq_table.pop(key)
            task.mq.state = QubitState.RAW
            proc.send_msg(ReserveMsg("RESERVE_ABORT", self.gid, key))

        self.log_debug("GROUP_DRAIN abort-tasks=%s uncancelable-tasks=%s", tasks_aborted, tasks_uncancelable)
        return self._drain_time(now)

    @override
    def assert_is_drained(self) -> None:
        assert len(self.oreq_table) == 0, f"{self}.oreq_table = {self.oreq_table}"

    @override
    def sync_external_enter(self) -> None:
        # Start reservations on all qubits that are in RAW state.
        for mq in self.qubits.values():
            if mq.state is QubitState.RAW:
                self.start_reservation(mq)

    @override
    def sync_external_exit(self) -> None:
        # Clear incomplete reservations.
        self.oreq_table.clear()

    def start_reservation(self, mq: MemoryQubit) -> None:
        """
        Start reservation on a qubit.
        """
        # Generate a unique reservation key to represent this entanglement generation protocol execution
        # and the entanglement, which is valid until the entanglement is released.
        mq.key = _AUTOID()
        mq.state = QubitState.ACTIVE

        # Remember the pending reservation.
        self.oreq_table[mq.key] = _EntangleTask(mq)
        self.log_debug("RESERVE_REQ key=%s mq-addr=%s", mq.key, mq.addr)

        # Transmit RESERVE_REQ message to the secondary node.
        self.proc.send_msg(ReserveMsg("RESERVE_REQ", self.gid, mq.key))

    def handle_reserve_res(self, req: ReserveMsg) -> None:
        proc = self.proc

        task = self.oreq_table.get(req.key, None)
        if task is None:
            self.log_debug("%s ignored reason=no-active-task", req)
            return
        assert task.k == -1, f"{task} reservation already accepted"

        # Mark the qubit as reserved.
        mq = task.mq
        assert mq.key == req.key
        mq.state = QubitState.RESERVED

        # Calculate which attempt would succeed and when would it finish.
        # TODO space out EPRs on a qchannel by attempt_interval or qchannel.bandwidth
        la = proc.ch.link_arch
        task.k = proc.ll._calc_attempt(proc.ch)
        task.t_reserve = proc.simulator.tc
        t_success = task.t_reserve + la.attempt_interval * (task.k - 1)
        task.t_notify_pri = t_success + la.d_notify_pri
        task.t_notify_2nd = t_success + la.d_notify_2nd
        t_finish = max(task.t_notify_pri, task.t_notify_2nd)

        # When using SYNC timing, if the successful attempt cannot finish within the current EXTERNAL phase,
        # the EPR is not scheduled because it cannot arrive in time.
        # Memory reset and table cleanups are performed by EXTERNAL phase exit trigger.
        if not proc.node.timing.is_external(t_finish):
            self.log_debug(
                "%s EPR_SKIP t_success=%s t_notify=%s,%s reason=beyond-external-phase",
                task,
                t_success,
                task.t_notify_pri,
                task.t_notify_2nd,
            )
            return

        # Create EPR and schedule success event.
        # EPR is created here instead of in .attempt_success() so that epr.name can be logged together.
        epr = la.make_epr(t_success, proc.node, proc.peer, key=task.key)
        task.success_event = LLEntangleSuccessEvent(self, task, epr, t=t_success)
        proc.simulator.sched(task.success_event)
        self.log_debug(
            "%s EPR_PREPARE epr=%s t_success=%s t_notify=%s,%s",
            task,
            epr.name,
            t_success,
            task.t_notify_pri,
            task.t_notify_2nd,
        )

    def attempt_success(self, task: "_EntangleTask", epr: Entanglement) -> None:
        """
        Start the successful attempt.

        Invoked by :py:class:`LLEntangleSuccessEvent`.
        """
        self.on_attempt_success(self.proc.ch, self.proc.peer, self.gid, epr)

        # Schedule EPR delivery events on both nodes.
        simulator = self.proc.simulator
        simulator.sched(LLNotifyPriEvent(self, task, epr, t=task.t_notify_pri))
        simulator.sched(LLNotify2ndEvent(task.key, epr, t=task.t_notify_2nd))

    def before_notify(self, task: "_EntangleTask") -> None:
        """
        Record a task before notifying primary node.

        Invoked by :py:class:`LLNotifyPriEvent`.
        """
        assert task.k >= 1, f"{task} reservation not accepted"
        del self.oreq_table[task.key]

        cnt = self.proc.cnt
        cnt.n_etg += 1
        cnt.n_attempts += task.k
        if cnt._history is not None:
            cnt._history.append(max(task.t_notify_pri.slot, task.t_notify_2nd.slot) - task.t_reserve.slot)

    @override
    def handle_release(self, event: QubitReleasedEvent) -> None:
        proc = self.proc
        self.log_debug("%s processed", event)
        if event.is_decoh:
            proc.cnt.n_decoh += 1
        if self.delete_event.get() is None and proc.node.timing.is_async():
            self.start_reservation(event.qubit)


@final
class LLEntangleSuccessEvent(Event):
    """
    Event to signal the start of the successful attempt.
    """

    def __init__(
        self,
        grp: LLGroupPri,
        task: "_EntangleTask",
        epr: Entanglement,
        *,
        t: Time,
    ):
        proc = grp.proc
        super().__init__(t, f"{proc.node.name}-{proc.peer.name}, key={task.key}")
        self.grp = grp
        self.task = task
        self.epr = epr

    @override
    def invoke(self) -> None:
        self.grp.attempt_success(self.task, self.epr)


@final
class LLNotifyPriEvent(LLNotifyEvent):
    """
    Event to notify the primary node about successful elementary entanglement.
    """

    def __init__(
        self,
        grp: LLGroupPri,
        task: "_EntangleTask",
        epr: Entanglement,
        *,
        t: Time,
    ):
        super().__init__(t, unwrap_cast(epr.src), unwrap_cast(epr.dst), task.key, epr)
        self.grp = grp
        self.task = task

    @override
    def invoke(self) -> None:
        self.grp.before_notify(self.task)
        return super().invoke()


class _EntangleTask:
    """
    Contextual information for creating one EPR pair.
    """

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
    t_notify_pri: Time = Time.SENTINEL
    """
    Time point to notify the primary node of successful entanglement.
    ``Time.SENTINEL`` means the reservation has not been accepted.
    """
    t_notify_2nd: Time = Time.SENTINEL
    """
    Time point to notify the secondary node of successful entanglement.
    ``Time.SENTINEL`` means the reservation has not been accepted.
    """
    success_event: LLEntangleSuccessEvent | None = None
    """
    Event to start successful attempt.
    ``None`` means the reservation has not been accepted or the notifications would exceed beyond EXTERNAL phase.
    """

    def __init__(self, mq: MemoryQubit):
        self.mq = mq

    @property
    def key(self) -> str:
        return unwrap(self.mq.key)

    def __repr__(self) -> str:
        return f"EntangleTask({self.key}, k={self.k}, t_reserve={self.t_reserve})"
