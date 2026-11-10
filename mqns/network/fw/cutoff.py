from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, final, override

from mqns.entity.memory import MemoryQubit, PathDirection
from mqns.network.fw.fib import FibPath
from mqns.network.fw.fw_module import ForwarderModule, fw_signaling_cmd_handler
from mqns.network.fw.message import CutoffDiscardMsg
from mqns.simulator import Event, Time
from mqns.utils import unwrap_cast

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class CutoffScheme(ForwarderModule, ABC):
    """
    EPR age cut-off scheme.

    This determines how ``PathInstructions.swap_cutoff`` is interpreted.
    """

    @classmethod
    def of(cls, fw: "Forwarder"):
        """
        Retrieve subclass instance from forwarder.
        """
        assert isinstance(fw.cutoff, cls)
        return fw.cutoff

    def initiate_discard(self, qubit: MemoryQubit, fp: FibPath, *, round=-1):
        """
        Discard a qubit that has exceeded cutoff time at the local forwarder.

        This is called by CutoffScheme subclass.

        Args:
            round: -1 for swap_cutoff; nonnegative number for purif round.
        """
        o_key = unwrap_cast(qubit.key)
        partner = unwrap_cast(qubit.partner)
        self.log_debug("local cutoff discard key=%s addr=%s round=%s partner=%s", o_key, qubit.addr, round, partner)

        # Inform SwapProc.
        if round == -1:
            self.fw.swap.handle_decohere(o_key)

        # Discard primary qubit.
        self.fw_cnt.increment_n_cutoff(round, True)
        self.fw.release_qubit(qubit, need_remove=True)

        # Ask partner to discard secondary qubit.
        msg: CutoffDiscardMsg = {
            "cmd": "CUTOFF_DISCARD",
            "path_id": fp.path_id,
            "key": partner.key,
            "round": round,
        }
        self.send_msg(partner.node, msg, fp)

    @fw_signaling_cmd_handler("CUTOFF_DISCARD")
    def handle_discard(self, msg: CutoffDiscardMsg, fp: FibPath):
        """
        Discard a qubit that has exceeded cutoff time at the remote forwarder.

        This is called by ProactiveForwarder upon receiving a CUTOFF_DISCARD message.
        """
        _ = fp
        o_key = msg["key"]
        round = msg["round"]

        # Inform SwapProc.
        if round == -1:
            self.fw.swap.handle_decohere(o_key)

        # Find qubit.
        qm_tuple = self.memory.read(o_key, remove=True)
        if qm_tuple is None:
            self.log_debug("remote cutoff discard key=%s not exist", o_key)
            return
        qubit, _ = qm_tuple
        self.log_debug("remote cutoff discard key=%s addr=%s round=%s", o_key, qubit.addr, round)

        # Discard secondary qubit.
        self.fw_cnt.increment_n_cutoff(round, False)
        self.fw.release_qubit(qubit)

    @abstractmethod
    def before_store_eligible(self, mq: MemoryQubit, dir: PathDirection, fp: FibPath | None) -> None:
        """
        Handle an ELIGIBLE qubit stored for future swapping.
        """

    @abstractmethod
    def before_swap(self, mq0: MemoryQubit, mq1: MemoryQubit, fp: FibPath | None) -> None:
        """
        Handle a pair of ELIGIBLE qubits before swapping.

        Args:
            mq0: Newly arrived qubit.
            mq1: Existing qubit chosen from memory.
        """


@final
class CutoffDiscardEvent(Event):
    def __init__(
        self,
        cutoff: CutoffScheme,
        qubit: MemoryQubit,
        fp: FibPath,
        *,
        t: Time,
        round: int,
        eligible_t: Time,
    ):
        super().__init__(t, f"addr={qubit.addr} key={qubit.key}")
        self.cutoff = cutoff
        self.qubit = qubit
        self.fp = fp
        self.round = round
        self.eligible_t = eligible_t

    @override
    def invoke(self) -> None:
        self.cutoff.initiate_discard(self.qubit, self.fp, round=self.round)


class CutoffSchemeWaitTimeCounters:
    def __init__(self):
        self.wait_values: list[int] | None = None
        """wait time values for waited qubits before swap, in time_slots"""

    def enable_collect_all(self) -> None:
        """Enable collecting all values for histogram generation."""
        self.wait_values = []


class CutoffSchemeWaitTime(CutoffScheme):
    """
    EPR age cut-off with individual wait-time budget.

    This cut-off scheme assigns a wait-time budget to each repeater node along a path.
    The controller provides these wait-time budgets in ``PathInstructions.swap_cutoff`` field.

    Each node individually tracks how long an EPR has been waiting in memory until it can be swapped.
    If an EPR has waited for more than the budget at this node, it cannot be used in a swap and would
    be released to make room for a new EPR.
    Note that ``swap_delay`` does not count against the wait budget.
    """

    _DIR_OFFSET: ClassVar = {
        PathDirection.L: -2,
        PathDirection.R: -1,
    }

    def __init__(self):
        self.cnt = CutoffSchemeWaitTimeCounters()

    @override
    def before_store_eligible(self, mq: MemoryQubit, dir: PathDirection, fp: FibPath | None) -> None:
        if not fp or (wait_budget := fp.swap_cutoff[2 * fp.own_idx + self._DIR_OFFSET[dir]]) is None:
            return

        now = self.simulator.tc
        deadline = now + wait_budget
        self.log_debug(
            "wait-time key=%s addr=%s path=%s(%s)%s budget=%s deadline=%s",
            mq.key,
            mq.addr,
            fp.path_id,
            fp.own_idx,
            dir.name,
            wait_budget,
            deadline,
        )
        self.simulator.sched(event := CutoffDiscardEvent(self, mq, fp, round=-1, eligible_t=now, t=deadline))
        mq.events.add(event)

    @override
    def before_swap(self, mq0: MemoryQubit, mq1: MemoryQubit, fp: FibPath | None) -> None:
        _ = fp
        assert mq0.events.get(CutoffDiscardEvent) is None

        event = mq1.events.discard(CutoffDiscardEvent)
        if event and self.cnt.wait_values is not None:
            self.cnt.wait_values.append((self.simulator.tc - event.eligible_t).slot)
