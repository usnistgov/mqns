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


@dataclass
class SwapArm:
    partner: QNode
    """Partner node."""
    index: int
    """Partner node index within FIB entry."""
    rank: int
    """Partner node swap rank within FIB entry."""

    qubit: MemoryQubit
    """Local qubit entangled with partner."""
    epr_name: str
    """Logical EPR name with partner."""
    phy_epr: Entanglement
    """Physical EPR with partner."""

    def __repr__(self) -> str:
        if self.epr_name == self.phy_epr.name:
            lepr = ""
        else:
            lepr = f"logical-epr={self.epr_name}, physical-"

        return (
            f"SwapArm({self.partner.name}@{self.index}, qubit={self.qubit.addr}, "
            f"{lepr}epr={self.phy_epr.name}@{cast(QNode, self.phy_epr.src).name}-{cast(QNode, self.phy_epr.dst).name})"
        )


class SwapTask:
    """
    Track the logical progress of a swap process within a swap group.

    An instance of ``SwapTask`` is lazily created when a node starts swapping two EPRs (either elementary EPR or
    completed EPR from lower-ranked swap group). The instance tracks how many nodes in the same ``FibSwapGroup``
    have completed their swaps, based on local operation outcomes and heralded information learned from other nodes
    within the swap group.
    """

    def __init__(self, fib_entry: FibEntry):
        self.path_id = fib_entry.path_id
        self.sg = FibSwapGroup.compute(fib_entry)

        self.o_complete = False
        """Is my own local swap completed?"""
        self.l_complete = self.sg.l_most
        """Is the chain to my left completed?"""
        self.r_complete = self.sg.r_most
        """Is the chain to my right completed?"""

        self.expiry: int | None = None
        """Time slot for group-wise EPR expiration, zero on swap failure anywhere."""

        self.l_sent = self.sg.l_most and not self.sg.l_herald
        """Is the left same-rank peer or higher-rank neighbor heralded?"""
        self.r_sent = self.sg.r_most and not self.sg.r_herald
        """Is the right same-rank peer or higher-rank neighbor heralded?"""

        self.lp_epr: str | None = None
        """Old EPR name toward left adjacent node."""
        self.rp_epr: str | None = None
        """Old EPR name toward right adjacent node."""

        self.lb_epr: str | None = None
        """Old EPR name known by leftmost node."""
        self.rb_epr: str | None = None
        """Old EPR name known by rightmost node."""

    def notify_local_swap(self, expiry: int, l_epr_name: str, r_epr_name: str):
        """
        Save local swap outcome.

        Args:
            expiry: Zero on swap failure, otherwise time slot for swapped EPR expiration time.
            l_epr_name: Left side EPR name.
            r_epr_name: Right side EPR name.

        Returns:
            [0]: Instruction to herald left node, if allowed.
            [1]: Instruction to herald right node, if allowed.
        """
        self._update_expiry(expiry)

        self.o_complete = True

        # These EPRs are known by the adjacent nodes.
        self.lp_epr = l_epr_name
        self.rp_epr = r_epr_name

        # If this node is the leftmost or rightmost with a group, these EPRs are known by the higher-ranked neighbors.
        if self.sg.l_most:
            self.lb_epr = l_epr_name
        if self.sg.r_most:
            self.rb_epr = r_epr_name

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
        self._update_expiry(su["expiry"])

        sg = self.sg
        swapper = su["o_node"]
        if not sg.l_most and swapper == sg.nodes[sg.own_idx - 1]:
            assert su["l_node"] == sg.l_neigh
            self.l_complete = True
            self.lb_epr = su["l_epr"]
        elif not sg.r_most and swapper == sg.nodes[sg.own_idx + 1]:
            assert su["r_node"] == sg.r_neigh
            self.r_complete = True
            self.rb_epr = su["r_epr"]
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
        if not self.l_sent and self.o_complete and self.r_complete:
            l_su = SwapUpdateMsg(
                cmd="SWAP_UPDATE",
                path_id=self.path_id,
                o_node=sg.nodes[sg.own_idx],
                l_node=sg.l_neigh if sg.l_most else sg.nodes[sg.own_idx - 1],
                r_node=sg.r_neigh,
                l_epr=cast(str, self.lb_epr if sg.l_most else self.lp_epr),
                r_epr=cast(str, self.rb_epr),
                expiry=cast(int, self.expiry),
            )
            self.l_sent = True

        r_su = None
        if not self.r_sent and self.o_complete and self.l_complete:
            r_su = SwapUpdateMsg(
                cmd="SWAP_UPDATE",
                path_id=self.path_id,
                o_node=sg.nodes[sg.own_idx],
                l_node=sg.l_neigh,
                r_node=sg.r_neigh if sg.r_most else sg.nodes[sg.own_idx + 1],
                l_epr=cast(str, self.lb_epr),
                r_epr=cast(str, self.rb_epr if sg.r_most else self.rp_epr),
                expiry=cast(int, self.expiry),
            )
            self.r_sent = True

        return l_su, r_su

    def __repr__(self) -> str:
        return (
            f"SwapTask(path_id={self.path_id}, group={self.sg.nodes}, "
            f"complete={self.l_complete and 'l' or '_'}{self.o_complete and 'o' or '_'}{self.r_complete and 'r' or '_'}, "
            f"sent={self.l_sent and 'l' or '_'}{self.r_sent and 'r' or '_'}, "
            f"epr={self.lb_epr},{self.lp_epr},{self.rp_epr},{self.rb_epr}, expiry={self.expiry})"
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

        self.waiting_su: dict[int, tuple[SwapUpdateMsg, FibEntry]] = {}
        """
        SwapUpdates received prior to QubitEntangledEvent.

        * Key: MemoryQubit addr.
        * Value: SwapUpdateMsg and FibEntry.
        """

        self.remote_swapped: dict[str, Entanglement] = {}
        """
        EPRs that have been swapped remotely but the SwapUpdateMsg have not arrived.

        * Key: Elementary EPR name, for which the node expects an incoming SWAP_UPDATE message.
        * Value: Physical EPR object.
        """

        self.task_by_epr: dict[str, SwapTask] = {}
        """
        SwapTask associated with an EPR.

        * Key: Elementary EPR name, for which the node expects an incoming SWAP_UPDATE message.
        * Value: SwapTask instance.
        """

        self.task_by_qubit: dict[int, SwapTask] = {}
        """
        SwapTask associated with a memory qubit.

        * Key: MemoryQubit addr, where the qubit is expected to be used in a subsequent ``.start()`` swapping.
        * Value: SwapTask instance.
        """

        self.alias_by_qubit: dict[int, str] = {}
        """
        Elementary EPR name known by the opposite partner on the EPR associated with a memory qubit.

        * Key: MemoryQubit addr, where the qubit has a stored EPR.
        * Value: Elementary EPR name known by the opposite partner of the stored EPR.
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
        for key in ("waiting_su", "remote_swapped", "task_by_epr", "task_by_qubit", "alias_by_qubit"):
            table = getattr(self, key)
            if n := len(table):
                log.warning(f"{self}: {key} is not empty: {table}")
                max_table_size = max(max_table_size, n)

        if self.table_leak_tol >= 0 and max_table_size > self.table_leak_tol:
            raise MemoryError("memory leak detected in data structures")

    def exit_internal_phase(self):
        """
        Called when the forwarder in SYNC timing mode exits an internal phase.
        """
        assert not self.waiting_su, f"waiting_su is not empty {self.waiting_su}"
        self.remote_swapped.clear()
        self.task_by_epr.clear()
        self.task_by_qubit.clear()
        self.alias_by_qubit.clear()

    def _herald(self, fib_entry: FibEntry, su: SwapUpdateMsg, dir: Literal["l", "r"]):
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
        assert mq0.state is QubitState.ELIGIBLE, f"unexpected state {mq0.state}"
        assert mq1.state is QubitState.ELIGIBLE, f"unexpected state {mq1.state}"

        # Set SWAPPING state, so that forwarder cannot start another swapping on the same qubit.
        mq0.state = QubitState.SWAPPING
        mq1.state = QubitState.SWAPPING

        # Schedule swap completion event.
        self.simulator.add_event(
            func_to_event(self.simulator.tc + self.delay.calculate(), self._s_finish, mq0, mq1, fib_entry, self.simulator.tc)
        )

    def _s_finish(self, mq0: MemoryQubit, mq1: MemoryQubit, fib_entry: FibEntry, swap_start: Time):
        """
        Complete swapping between two memory qubits.

        This is scheduled by ``.start()`` after Bell-State Analyzer delay.
        """

        # Read both qubits and remove them from memory.
        #
        # If either qubit is no longer in SWAPPING state, it implies that a SWAP_UPDATE message arrived that informs
        # a failure for a parallel swap at a remote node, which caused the qubit to be released.
        # In this case, the local swapping is treated as aborted, and the other involved qubit is released.
        arm0 = self._s_get_arm(mq0, fib_entry)
        arm1 = self._s_get_arm(mq1, fib_entry)
        if arm0 is None or arm1 is None:
            log.debug(f"{self}: SWAP ABORT | {mq0} x {mq1}")
            for target in arm0, arm1:
                if target:
                    self.fw.release_qubit(target.qubit)
                    # TODO construct HeraldInformation and send SWAP_UPDATE
                    # self._send_su(fib_entry, target, "", target.epr.name, None)
            return

        # One of these qubits must be entangled with a partner node to the left of the current node.
        # This is determined by epr.dst==self.node condition, because LinkLayer establishes elementary
        # entanglements from left to right, and swapping maintains this condition.
        # This qubit and related objects are assigned to `prev`.
        #
        # Likewise, the other qubit entangled with a partner node to the right is assigned to `next`.
        prev, next = (arm0, arm1) if arm0.index < arm1.index else (arm1, arm0)

        # Save ch_index metadata field onto elementary EPR.
        if not prev.phy_epr.orig_eprs:
            assert prev.phy_epr.name == prev.epr_name
            prev.phy_epr.ch_index = fib_entry.own_idx - 1
        if not next.phy_epr.orig_eprs:
            assert next.phy_epr.name == next.epr_name
            next.phy_epr.ch_index = fib_entry.own_idx

        # Attempt the swap.
        new_epr, local_success = Entanglement.swap(prev.phy_epr, next.phy_epr, now=swap_start, ps=self.ps, error=self.error)
        log.debug(
            f"{self}: SWAP {'SUCC' if local_success else 'FAILED'} rank={fib_entry.own_swap_rank} | {prev} x {next} = {new_epr}"
        )

        # Release consumed qubits.
        self.fw.release_qubit(prev.qubit)
        self.fw.release_qubit(next.qubit)

        # Update physical swap counters.
        if local_success:
            self.fw.cnt.n_swapped += 1

            # Inform multiplexing scheme.
            # TODO audit whether MuxScheme would access unheralded information
            self.mux.swapping_succeeded(prev.phy_epr, next.phy_epr, new_epr)
        else:
            self.fw.cnt.n_swap_fail += 1

        # Retrieve SwapTask and record local swap outcome.
        task, task_from = self._s_get_task(fib_entry, prev, next)
        # XXX new_epr.decohere_time.time_slot may access unheralded information.
        l_su, r_su = task.notify_local_swap(
            new_epr.decohere_time.time_slot if local_success else 0,
            prev.epr_name,
            next.epr_name,
        )

        # Deposit physical swap result.
        self._s_physical_deposit(new_epr)

        # Sending heralding if allowed.
        if l_su:
            self._herald(fib_entry, l_su, "l")
        if r_su:
            self._herald(fib_entry, r_su, "r")

        # Store SwapTask if own node expect heralding from left/right.
        task_saved: list[str] = []
        if not task.l_complete:
            self.task_by_epr[prev.epr_name] = task
            task_saved.append(f"task_by_epr[prev.epr_name:={prev.epr_name}]")
        if not task.r_complete:
            self.task_by_epr[next.epr_name] = task
            task_saved.append(f"task_by_epr[next.epr_name:={next.epr_name}]")
        log.debug(f"{self}: {task} retrieved-from={task_from} saved-at={task_saved}")

    def _s_get_arm(self, mq: MemoryQubit, fib_entry: FibEntry) -> SwapArm | None:
        """Retrieve information related to a memory qubit during swapping."""

        if mq.state is not QubitState.SWAPPING:
            return None

        qubit, epr = self.memory.read(mq.addr, has=self.epr_type, remove=True)

        if epr.dst is self.node:
            partner = epr.src
        elif epr.src is self.node:
            partner = epr.dst
        else:
            raise RuntimeError(f"{self}: node not in {epr} stored at {mq}")
        assert partner is not None

        epr_name = self.alias_by_qubit.pop(qubit.addr, epr.name)

        while True:
            phy = self.remote_swapped.pop(epr.name, None)
            if phy:
                epr = phy
            else:
                break

        return SwapArm(partner, *fib_entry.find_index_and_swap_rank(partner.name), qubit, epr_name, epr)

    def _s_get_task(self, fib_entry: FibEntry, prev: SwapArm, next: SwapArm) -> tuple[SwapTask, str]:
        if t := self.task_by_epr.pop(prev.epr_name, None):
            return t, f"task_by_epr[prev.epr_name:={prev.epr_name}]"
        if t := self.task_by_epr.pop(next.epr_name, None):
            return t, f"task_by_epr[next.epr_name:={next.epr_name}]"
        if t := self.task_by_qubit.pop(prev.qubit.addr, None):
            return t, f"task_by_qubit[prev.qubit.addr:={prev.qubit.addr}]"
        if t := self.task_by_qubit.pop(next.qubit.addr, None):
            return t, f"task_by_qubit[next.qubit.addr:={next.qubit.addr}]"
        return SwapTask(fib_entry), "constructor"

    def _s_physical_deposit(self, new_epr: Entanglement) -> None:
        deposit_at: list[str] = []
        assert new_epr.orig_eprs
        for pos, attr in (0, "src"), (-1, "dst"):
            target = cast(QNode, getattr(new_epr, attr))
            old_epr = new_epr.orig_eprs[pos]
            target.get_app(type(self.fw)).swap.remote_swapped[old_epr.name] = new_epr
            deposit_at.append(f"{target.name}.remote_swapped[{old_epr.name}]")
        log.debug(f"{self}: physical deposit at {deposit_at}")

    def pop_waiting_su(self, qubit: MemoryQubit):
        """
        Invoked by ``Forwarder.qubit_is_entangled()`` after QubitEntangledEvent to process buffered SWAP_UPDATE.

        It's possible for SWAP_UPDATE to arrive in the same time slot as QubitEntangledEvent, for example in
        S-R-D linear topology, when node R performs a swap as soon as it is notified about R-D entanglement,
        before D is notified. This cannot be resolved with ``Event.priority`` mechanism because R and D may be
        notified at different times depending on the link architecture.
        """
        su_args = self.waiting_su.pop(qubit.addr, None)
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
            # Own node is to the right of the swapper, so that swapper's r_epr is in own memory.
            old_epr_name = msg["r_epr"]
        else:
            old_epr_name = msg["l_epr"]

        # Defer after QubitEntangledEvent for the qubit is processed.
        if (qubit_pair := self.memory.read(old_epr_name)) and (qubit := qubit_pair[0]).state is QubitState.ENTANGLED0:
            self.waiting_su[qubit.addr] = (msg, fib_entry)
            return

        # If the swapper has a lower rank, it indicates the completion of a lower-ranked swap task.
        # Save its outcome into memory, so that this node can start purification and own-rank swapping,
        if swapper_rank < fib_entry.own_swap_rank:
            self._u_lower(msg, fib_entry, qubit_pair, old_epr_name, swapper_idx, swapper_rank)
            return

        # If the swapper has the same rank, it is part of the swap task that is potentially parallel.
        assert swapper_rank == fib_entry.own_swap_rank
        self._u_same(msg, fib_entry, qubit_pair, old_epr_name)

    def _u_lower(
        self,
        msg: SwapUpdateMsg,
        fib_entry: FibEntry,
        qubit_pair: tuple[MemoryQubit, QuantumModel | None] | None,
        old_epr_name: str,
        swapper_idx: int,
        swapper_rank: int,
    ) -> None:
        """Process SWAP_UPDATE sent from a lower-ranked node."""
        self.fw.cnt.n_su_lower += 1

        # Retrieve qubit and new physical EPR.
        assert qubit_pair, f"qubit not found for {old_epr_name}"
        qubit, _ = qubit_pair
        new_epr = self.remote_swapped.pop(old_epr_name)

        # Verify that the new physical EPR matches the heralded EPR segment.
        # This logic only supports ASAP parallel swap at the lowest rank.
        # Having parallel swap at any higher rank would break the assertion, because the physical EPR
        # may have been swapped by a peer when own node processes SWAP_UPDATE.
        if swapper_idx < fib_entry.own_idx:
            assert cast(QNode, new_epr.src).name == (partner := msg["l_node"])
            assert new_epr.dst is self.node
            alias = msg["l_epr"]
        else:
            assert cast(QNode, new_epr.dst).name == (partner := msg["r_node"])
            assert new_epr.src is self.node
            alias = msg["r_epr"]

        # If the lower-ranked swap failed or the new EPR has decohered, release the qubit.
        # We can only make this determination based on heralded expiration time.
        if msg["expiry"] <= self.simulator.tc.time_slot:
            log.debug(f"{self}: NEW EPR {new_epr} decohered during SU transmissions")
            self.fw.release_qubit(qubit, need_remove=True)
            return

        # Store new EPR.
        self.memory.write(qubit.addr, new_epr, replace=True)
        if fib_entry.own_idx not in (0, len(fib_entry.route) - 1):
            self.alias_by_qubit[qubit.addr] = alias

        log.debug(
            f"{self}: segment {cast(QNode, new_epr.src).name}-{cast(QNode, new_epr.dst).name} "
            f"swap completed for rank {swapper_rank}"
        )

        # Progress toward purification and this-rank swap.
        qubit.purif_rounds = 0
        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, self.network.get_node(partner))

    def _u_same(
        self,
        msg: SwapUpdateMsg,
        fib_entry: FibEntry,
        qubit_pair: tuple[MemoryQubit, QuantumModel | None] | None,
        old_epr_name: str,
    ) -> None:
        """Process SWAP_UPDATE sent from a same-ranked node."""
        self.fw.cnt.n_su_same += 1

        # Retrieve SwapTask and record heralded swap outcome.
        task, task_from = self._u_get_task(fib_entry, old_epr_name)
        l_su, r_su = task.notify_remote_swap(msg)

        # Sending heralding if allowed.
        if l_su:
            self._herald(fib_entry, l_su, "l")
        if r_su:
            self._herald(fib_entry, r_su, "r")

        # Store SwapTask if own node has not swapped.
        task_saved: list[str] = []
        if not task.o_complete:
            assert qubit_pair is not None, f"qubit not found for {old_epr_name} | {task} retrieved-from={task_from}"
            qubit = qubit_pair[0]
            self.task_by_qubit[qubit.addr] = task
            task_saved.append(f"task_by_qubit[qubit.addr:={qubit.addr}]")

        log.debug(f"{self}: {task} retrieved-from={task_from} saved-at={task_saved}")

    def _u_get_task(self, fib_entry: FibEntry, old_epr_name: str) -> tuple[SwapTask, str]:
        if t := self.task_by_epr.pop(old_epr_name, None):
            return t, f"task_by_epr[old_epr_name:={old_epr_name}]"
        return SwapTask(fib_entry), "constructor"

    def __repr__(self) -> str:
        return Application.__repr__(self)
