from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from mqns.entity.memory import MemoryQubit, QuantumMemory, QubitState
from mqns.entity.node import Application, QNode
from mqns.models.core import QuantumModel
from mqns.models.delay import DelayModel
from mqns.models.epr import Entanglement
from mqns.models.error import ErrorModel
from mqns.network.fw.fib import FibEntry, FibSwapGroup
from mqns.network.fw.message import SwapUpdateMsg
from mqns.network.fw.mux import MuxScheme
from mqns.network.network import QuantumNetwork
from mqns.simulator import Simulator, Time, func_to_event
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


def _qubit_key(mq: MemoryQubit) -> str:
    assert mq.key, f"missing reservation key in {mq}"
    return mq.key


@dataclass
class SwapArm:
    mq: MemoryQubit
    """Local qubit entangled with partner."""
    o_key: str
    """Local qubit reservation key."""
    p_key: str
    """Partner qubit reservation key."""

    def __repr__(self) -> str:
        return f"SwapArm(qubit={self.mq.addr}, o-key={self.o_key}, p-key={self.p_key})"


class SwapTask:
    """
    Track the logical progress of a swap process within a swap group.

    An instance of ``SwapTask`` is lazily created when a node starts swapping two EPRs (either elementary EPR or
    completed EPR from lower-ranked swap group). The instance tracks how many nodes in the same ``FibSwapGroup``
    have completed their swaps, based on local operation outcomes and heralded information learned from other nodes
    within the swap group.
    """

    o_started = False
    """Is my own local swap started?"""
    o_complete = False
    """Is my own local swap completed?"""
    l_complete: bool
    """Is the chain to my left completed?"""
    r_complete: bool
    """Is the chain to my right completed?"""

    expiry: int | None = None
    """Time slot for group-wise EPR expiration, zero on swap failure anywhere."""

    l_sent = False
    """Is the left same-rank peer or higher-rank neighbor heralded?"""
    r_sent = False
    """Is the right same-rank peer or higher-rank neighbor heralded?"""

    # Default values in la_key,ra_key,lb_key,rb_key are only used when heralding failure to the opposite direction.

    la_key = "SwapTask.la_key=UNKNOWN"
    """Qubit reservation key between own node and ``sg.l_adj``."""
    ra_key = "SwapTask.ra_key=UNKNOWN"
    """Qubit reservation key between own node and ``sg.r_adj``."""

    lb_key = "SwapTask.lb_key=UNKNOWN"
    """Qubit reservation key between ``sg.nodes[0]`` and ``sg.l_neigh``."""
    rb_key = "SwapTask.rb_key=UNKNOWN"
    """Qubit reservation key between ``sg.nodes[-1]`` and ``sg.l_neigh``."""

    def __init__(self, fib_entry: FibEntry):
        self.path_id = fib_entry.path_id
        self.sg = FibSwapGroup.compute(fib_entry)
        self.l_complete = self.sg.l_most
        self.r_complete = self.sg.r_most

    def begin_local_swap(self, la_key: str, ra_key: str) -> None:
        """
        Record start of local swap.

        Args:
            la_key: Qubit reservation key known by left adjacent node.
            ra_key: Qubit reservation key known by right adjacent node.
        """

        self.o_started = True

        # These EPRs are known by both own node and the adjacent nodes.
        self.la_key = la_key
        self.ra_key = ra_key

        # If this node is the leftmost or rightmost with a group, these EPRs are known by the higher-ranked neighbors.
        if self.sg.l_most:
            self.lb_key = self.la_key
        if self.sg.r_most:
            self.rb_key = self.ra_key

    def end_local_swap(self, expiry: int):
        """
        Save local swap outcome.

        Args:
            expiry: Zero on swap failure, otherwise time slot for swapped EPR expiration time.

        Returns:
            [0]: Instruction to herald left node, if allowed.
            [1]: Instruction to herald right node, if allowed.
        """
        assert self.o_started, "{self}: end_local_swap without start_local_swap"

        self._update_expiry(expiry)

        self.o_complete = True

        # If local swap failed, there's no need to wait for heralding.
        if expiry == 0:
            self.l_complete = True
            self.r_complete = True

        return self._check_triggers()

    def notify_remote_swap(self, su: SwapUpdateMsg):
        """
        Save heralded swap outcome.

        Args:
            su: SWAP_UPDATE message.

        Returns:
            [0]: Instruction to herald left node, if allowed.
            [1]: Instruction to herald right node, if allowed.
        """
        expiry = su["expiry"]
        self._update_expiry(expiry)

        sg = self.sg
        swapper = su["o_node"]
        if swapper == sg.l_adj:
            # Message came from left peer.
            assert su["l_node"] == sg.l_neigh
            self.l_complete = True
            self.lb_key = su["l_key"]
            if expiry == 0:
                # Left peer does not need to hear about the failure they just told us.
                self.l_sent = True
        elif swapper == sg.r_adj:
            # Message came from right peer.
            assert su["r_node"] == sg.r_neigh
            self.r_complete = True
            self.rb_key = su["r_key"]
            if expiry == 0:
                # Right peer does not need to hear about the failure they just told us.
                self.r_sent = True
        else:
            raise RuntimeError(f"SwapGroup({sg.nodes},{sg.own_idx}) received swap outcome from unexpected node {swapper}")

        return self._check_triggers()

    def _update_expiry(self, expiry: int) -> None:
        if self.expiry is None:
            self.expiry = expiry
        else:
            self.expiry = min(self.expiry, expiry)

    def _check_triggers(self) -> tuple[SwapUpdateMsg | None, SwapUpdateMsg | None]:
        sg = self.sg

        l_su = None
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
            l_su = SwapUpdateMsg(
                cmd="SWAP_UPDATE",
                path_id=self.path_id,
                o_node=sg.nodes[sg.own_idx],
                l_node=sg.l_adj,
                r_node=sg.r_neigh,
                l_key=self.lb_key if sg.l_most else self.la_key,
                r_key=self.rb_key,
                expiry=cast(int, self.expiry),
            )
            self.l_sent = True

        r_su = None
        if not self.r_sent and (
            self.o_started if self.expiry == 0 else (self.l_complete and self.o_complete and sg.dir in ("r", "b"))
        ):
            r_su = SwapUpdateMsg(
                cmd="SWAP_UPDATE",
                path_id=self.path_id,
                o_node=sg.nodes[sg.own_idx],
                l_node=sg.l_neigh,
                r_node=sg.r_adj,
                l_key=self.lb_key,
                r_key=self.rb_key if sg.r_most else self.ra_key,
                expiry=cast(int, self.expiry),
            )
            self.r_sent = True

        return l_su, r_su

    def __repr__(self) -> str:
        return (
            f"SwapTask(path_id={self.path_id}, group={self.sg.nodes}, "
            f"complete={self.l_complete and 'l' or '_'}"
            f"{self.o_complete and 'o' or (self.o_started and 'O' or '_')}"
            f"{self.r_complete and 'r' or '_'}, "
            f"sent={self.l_sent and 'l' or '_'}{self.r_sent and 'r' or '_'}, "
            f"qubit-key={self.lb_key},{self.la_key},{self.ra_key},{self.rb_key}, expiry={self.expiry})"
        )


class ForwarderSwapProc:
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

    fw: "Forwarder"
    simulator: Simulator
    epr_type: type[Entanglement]
    network: QuantumNetwork
    node: QNode
    memory: QuantumMemory
    mux: MuxScheme

    def __init__(self, *, ps: float, delay: DelayModel, error: ErrorModel):
        self.ps = ps
        """Probability of successful entanglement swapping."""
        assert 0.0 <= self.ps <= 1.0
        self.delay = delay
        """Swapping delay model."""
        self.error = error
        """Swapping error model."""

        self.waiting_su: dict[str, tuple[SwapUpdateMsg, FibEntry]] = {}
        """
        SwapUpdates received prior to QubitEntangledEvent.

        * Key: MemoryQubit reservation key.
        * Value: SwapUpdateMsg and FibEntry.
        """

        self.remote_swapped: dict[str, Entanglement] = {}
        """
        EPRs that have been swapped remotely but the SwapUpdateMsg have not arrived.

        * Key: MemoryQubit reservation key, for which the node expects an incoming SWAP_UPDATE message.
        * Value: Physical EPR object.
        """

        self.task_by_qubit: dict[str, SwapTask] = {}
        """
        SwapTask associated with a memory qubit.

        * Key: MemoryQubit reservation key.
        * Value: SwapTask instance.
        """

    def install(self, fw: "Forwarder"):
        self.fw = fw
        self.simulator = fw.simulator
        self.epr_type = fw.epr_type
        self.network = fw.network
        self.node = fw.node
        self.memory = fw.memory
        self.mux = fw.mux

        if self.simulator.te:
            event = func_to_event(self.simulator.te, self.check_table_leak)
            event.priority = 0x1FFFFFFF
            self.simulator.add_event(event)

    def check_table_leak(self) -> None:
        """Check for memory leak in internal data structures."""
        max_table_size = 0
        for attr in ("waiting_su", "remote_swapped", "task_by_qubit"):
            table = getattr(self, attr)
            if n := len(table):
                log.warning(f"{self}: {attr} is not empty: {table}")
                max_table_size = max(max_table_size, n)

        if self.table_leak_tol >= 0 and max_table_size > self.table_leak_tol:
            raise MemoryError("memory leak detected in data structures")

    def exit_internal_phase(self) -> None:
        """
        Called when the forwarder in SYNC timing mode exits an internal phase.
        """
        assert not self.waiting_su, f"waiting_su is not empty {self.waiting_su}"
        self.remote_swapped.clear()
        self.task_by_qubit.clear()

    def _herald(self, fib_entry: FibEntry, su: SwapUpdateMsg, dir: Literal["l", "r"]) -> None:
        """
        Send heralding message.

        Args:
            fib_entry: FIB entry of the path.
            su: SWAP_UPDATE message.
            dir: Message direction, sending to ``su["l_node"]`` or ``su["r_node"]`.
        """
        target = self.network.get_node(su[cast(Any, f"{dir}_node")])
        self.fw.send_msg(target, su, fib_entry)

    def start(self, mq0: MemoryQubit, mq1: MemoryQubit, fib_entry: FibEntry):
        """
        Start swapping between two memory qubits.

        Args:
            mq0: First qubit, must be in ELIGIBLE state.
            mq1: Second qubit, must be in ELIGIBLE state and come from a different qchannel.
            fib_entry: FIB entry.
        """
        assert mq0.addr != mq1.addr
        assert mq0.qchannel is not mq1.qchannel

        # Retrieve both qubits and determine directions.
        arms = self._s_get_arms(mq0, mq1)
        prev, next = arms

        # Record local swap start in SwapTask.
        task, task_from = self._s_get_task(fib_entry, arms)
        task.begin_local_swap(prev.p_key, next.p_key)
        task_saved = self._s_put_task(task, arms)

        # Schedule swap completion event.
        now = self.simulator.tc
        finish_time = now + self.delay.calculate()
        self.simulator.add_event(func_to_event(finish_time, self._s_finish, arms, fib_entry, now, task))

        log.debug(f"{self}: SWAP_START {task} retrieved-from={task_from} saved-at={task_saved} finish-time={finish_time}")

    def _s_get_arms(self, mq0: MemoryQubit, mq1: MemoryQubit) -> Sequence[SwapArm]:
        arms: MutableSequence[SwapArm | None] = [None, None]

        for mq in mq0, mq1:
            # Retrieve qubit.
            _, epr = self.memory.read(mq.addr, has=self.epr_type)
            o_key = _qubit_key(mq)
            p_key = o_key if mq.partner is None else mq.partner[1]

            # Set SWAPPING state, so that forwarder cannot start another swapping on the same qubit.
            # The ALLOWED_STATE_TRANSITIONS matrix verifies existing state is ELIGIBLE.
            mq.state = QubitState.SWAPPING

            # Determine direction.
            if epr.dst is self.node:
                idx = 0
            elif epr.src is self.node:
                idx = 1
            else:
                raise RuntimeError(f"{self}: node not in {epr} stored at {mq}")

            # Save to destination array.
            # Ensure each arm has a different direction.
            assert arms[idx] is None
            arms[idx] = SwapArm(mq, o_key, p_key)

        return cast(Sequence[SwapArm], arms)

    def _s_get_task(
        self, fib_entry: FibEntry, arms: Sequence[SwapArm], task_via_event: SwapTask | None = None
    ) -> tuple[SwapTask, str]:
        task: SwapTask | None = None
        task_from: list[str] = []
        for arm in arms:
            if t := self.task_by_qubit.pop(arm.o_key, None):
                task = t
                task_from.append(f"task_by_qubit[{arm.o_key}]")
        if task:
            return task, ",".join(task_from)

        if task_via_event:
            return task_via_event, "event"
        return SwapTask(fib_entry), "constructor"

    def _s_put_task(self, task: SwapTask, arms: Sequence[SwapArm]) -> list[str]:
        task_saved: list[str] = []
        for i, dir_complete in enumerate((task.l_complete, task.r_complete)):
            if dir_complete:
                continue
            arm = arms[i]
            self.task_by_qubit[arm.o_key] = task
            task_saved.append(f"task_by_qubit[{arm.o_key}]")
        return task_saved

    def _s_finish(self, arms: list[SwapArm], fib_entry: FibEntry, swap_start: Time, task_via_event: SwapTask):
        """
        Complete swapping between two memory qubits.

        This is scheduled by ``.start()`` after Bell-State Analyzer delay.
        """

        prev, next = arms

        # Retrieve physical EPRs.
        # Save ch_index metadata field onto elementary EPR.
        local_expiry: MutableSequence[int] = []
        phy_eprs: MutableSequence[Entanglement] = []
        for i, arm in enumerate(arms):
            assert arm.mq.state is QubitState.SWAPPING, f"unexpected state {arm.mq.state}"
            _, epr = self.memory.read(arm.mq.addr, has=self.epr_type, remove=True)
            local_expiry.append(epr.decohere_time.time_slot)
            phy = self.remote_swapped.pop(arm.o_key, epr)
            if not phy.orig_eprs:
                phy.ch_index = fib_entry.own_idx - 1 + i
            phy_eprs.append(phy)

        # Attempt the swap.
        # Memory error model is applied as of the swap start time.
        new_epr, local_success = Entanglement.swap(*phy_eprs, now=swap_start, ps=self.ps, error=self.error)
        log.debug(
            f"{self}: SWAP_{'SUCC' if local_success else 'FAIL'} rank={fib_entry.own_swap_rank} | {prev} x {next} = {new_epr}"
        )

        # Release consumed qubits.
        for arm in arms:
            self.fw.release_qubit(arm.mq)

        # Update physical swap counters.
        if local_success:
            self.fw.cnt.n_swapped += 1

            # Inform multiplexing scheme.
            # TODO audit whether MuxScheme would access unheralded information
            self.mux.swapping_succeeded(phy_eprs[0], phy_eprs[1], new_epr)
        else:
            self.fw.cnt.n_swap_fail += 1

        # Record local swap outcome in SwapTask.
        task, task_from = self._s_get_task(fib_entry, arms, task_via_event)
        l_su, r_su = task.end_local_swap(min(local_expiry) if local_success else 0)
        task_saved = self._s_put_task(task, arms)

        # Deposit physical swap result.
        self._s_physical_deposit(new_epr)

        # Sending heralding if allowed.
        if l_su:
            self._herald(fib_entry, l_su, "l")
        if r_su:
            self._herald(fib_entry, r_su, "r")

        log.debug(f"{self}: SWAP_FINISH {task} retrieved-from={task_from} saved-at={task_saved}")

    def _s_physical_deposit(self, new_epr: Entanglement) -> None:
        if new_epr.is_decohered:
            log.debug(f"{self}: physical deposit skipped reason=DECOHERED")
            return

        deposit_at: list[str] = []
        assert new_epr.orig_eprs
        for attr, key_i in ("src", 0), ("dst", 1):
            target = cast(QNode, getattr(new_epr, attr))
            key = new_epr.mem_keys[key_i]
            target.get_app(type(self.fw)).swap.remote_swapped[key] = new_epr
            deposit_at.append(f"{target.name}.remote_swapped[{key}]")
        log.debug(f"{self}: physical deposit at {', '.join(deposit_at)}")

    def pop_waiting_su(self, qubit: MemoryQubit):
        """
        Invoked by ``Forwarder.qubit_is_entangled()`` after QubitEntangledEvent to process buffered SWAP_UPDATE.

        It's possible for SWAP_UPDATE to arrive in the same time slot as QubitEntangledEvent, for example in
        S-R-D linear topology, when node R performs a swap as soon as it is notified about R-D entanglement,
        before D is notified. This cannot be resolved with ``Event.priority`` mechanism because R and D may be
        notified at different times depending on the link architecture.
        """
        su_args = self.waiting_su.pop(_qubit_key(qubit), None)
        if (
            qubit.state is not QubitState.RELEASE  # qubit was released due to uninstalled path
            and su_args
        ):
            self.handle_update(*su_args)

    def handle_update(self, msg: SwapUpdateMsg, fib_entry: FibEntry) -> None:
        """
        Process an SWAP_UPDATE signaling message.

        Args:
            msg: The SWAP_UPDATE message.
            fib_entry: FIB entry associated with path_id in the message.
        """
        if not self.node.timing.is_internal():
            log.debug(f"{self}: INT phase is over -> stop swaps")
            return

        swapper_idx, swapper_rank = fib_entry.find_index_and_swap_rank(msg["o_node"])
        if swapper_idx < fib_entry.own_idx:
            # Own node is to the right of the swapper, so that swapper's r_key is in own memory.
            o_key = msg["r_key"]
        else:
            o_key = msg["l_key"]

        # Defer after QubitEntangledEvent for the qubit is processed.
        if (qubit_pair := self.memory.read(o_key)) and qubit_pair[0].state is QubitState.ENTANGLED0:
            self.waiting_su[o_key] = (msg, fib_entry)
            return

        # If the swapper has a lower rank, it indicates the completion of a lower-ranked swap task.
        # Save its outcome into memory, so that this node can start purification and own-rank swapping,
        if swapper_rank < fib_entry.own_swap_rank:
            self._u_lower(msg, fib_entry, qubit_pair, o_key, swapper_idx, swapper_rank)
            return

        # If the swapper has the same rank, it is part of the swap task that is potentially parallel.
        # qubit_pair would be None, if own node has already swapped.
        assert swapper_rank == fib_entry.own_swap_rank
        self._u_same(msg, fib_entry, qubit_pair, o_key)

    def _u_lower(
        self,
        msg: SwapUpdateMsg,
        fib_entry: FibEntry,
        qubit_pair: tuple[MemoryQubit, QuantumModel | None] | None,
        qubit_key: str,
        swapper_idx: int,
        swapper_rank: int,
    ) -> None:
        """Process SWAP_UPDATE sent from a lower-ranked node."""
        self.fw.cnt.n_su_lower += 1

        # Retrieve qubit and new physical EPR.
        assert qubit_pair, f"qubit not found for {qubit_key}"
        qubit = qubit_pair[0]
        new_phy = self.remote_swapped.pop(qubit_key, None)

        # If the lower-ranked swap failed or the new EPR has decohered, release the qubit.
        # We can only make this determination based on heralded expiration time.
        if (expiry := msg["expiry"]) <= self.simulator.tc.time_slot:
            if expiry == 0:
                log.debug(f"{self}: releasing qubit {qubit.addr} reason=lower-swap-failure key={qubit_key} | {new_phy}")
            else:
                log.debug(
                    f"{self}: releasing qubit {qubit.addr} reason=lower-expiry expiry={self.simulator.time(time_slot=expiry)} "
                    f"key={qubit_key} | {new_phy}"
                )
            self.fw.release_qubit(qubit, need_remove=True)
            return
        assert new_phy is not None, f"new_phy not found for {qubit}"

        # Verify that the new physical EPR matches the heralded EPR segment.
        # This logic only supports ASAP parallel swap at the lowest rank.
        # Having parallel swap at any higher rank would break the assertion, because the physical EPR
        # may have been swapped by a peer when own node processes SWAP_UPDATE.
        if swapper_idx < fib_entry.own_idx:
            assert (partner := cast(QNode, new_phy.src)).name == msg["l_node"]
            assert new_phy.dst is self.node
            p_key = msg["l_key"]
        else:
            assert (partner := cast(QNode, new_phy.dst)).name == msg["r_node"]
            assert new_phy.src is self.node
            p_key = msg["r_key"]

        # Store new EPR.
        qubit.partner = partner, p_key
        self.memory.write(qubit.addr, new_phy, replace=True)

        log.debug(
            f"{self}: segment {cast(QNode, new_phy.src).name}-{cast(QNode, new_phy.dst).name} "
            f"swap completed for rank {swapper_rank}"
        )

        # Progress toward purification and this-rank swap.
        qubit.purif_rounds = 0
        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, partner)

    def _u_same(
        self,
        msg: SwapUpdateMsg,
        fib_entry: FibEntry,
        qubit_pair: tuple[MemoryQubit, QuantumModel | None] | None,
        qubit_key: str,
    ) -> None:
        """Process SWAP_UPDATE sent from a same-ranked node."""
        self.fw.cnt.n_su_same += 1

        # Retrieve SwapTask.
        task, task_from = self._u_get_task(fib_entry, qubit_key)

        # If qubit does not exist, it means own node had swap failure previously and already notified both sides,
        # but the remote swap occurred while the outgoing SWAP_UPDATE is still in flight.
        if not task.o_complete and qubit_pair is None:
            log.debug(f"{self}: SWAP_UPDATE_SAME {task} retrieved-from={task_from} dropped reason=previous-swap-failure")
            assert task_from == "constructor"
            return

        # Record heralded swap outcome in SwapTask.
        l_su, r_su = task.notify_remote_swap(msg)
        task_saved = None
        if not task.o_complete:
            self.task_by_qubit[qubit_key] = task
            task_saved = f"task_by_qubit[{qubit_key}]"

        # Sending heralding if allowed.
        if l_su:
            self._herald(fib_entry, l_su, "l")
        if r_su:
            self._herald(fib_entry, r_su, "r")

        log.debug(f"{self}: SWAP_UPDATE_SAME {task} retrieved-from={task_from} saved-at={task_saved}")

    def _u_get_task(self, fib_entry: FibEntry, qubit_key: str) -> tuple[SwapTask, str]:
        if t := self.task_by_qubit.pop(qubit_key, None):
            return t, f"task_by_qubit[qubit_key:={qubit_key}]"
        return SwapTask(fib_entry), "constructor"

    def __repr__(self) -> str:
        return Application.__repr__(self)
