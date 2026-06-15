from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final, override

from mqns.entity.memory import MemoryQubit, PathDirection
from mqns.entity.node import QNode
from mqns.network.fw.fib import FibEntry
from mqns.network.fw.message import CutoffDiscardMsg
from mqns.simulator import Event, Simulator, Time
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class CutoffScheme(ABC):
    """
    EPR age cut-off scheme.

    This determines how ``PathInstructions.swap_cutoff`` is interpreted.
    """

    fw: "Forwarder"
    simulator: Simulator
    node: QNode

    def __init__(self, name: str):
        self.name = name
        """Scheme name."""

    def __repr__(self):
        return f"<{self.name}>"

    def install(self, fw: "Forwarder"):
        self.fw = fw
        self.simulator = fw.simulator
        self.node = fw.node

    @classmethod
    def of(cls, fw: "Forwarder"):
        """
        Retrieve subclass instance from forwarder.
        """
        assert isinstance(fw.cutoff, cls)
        return fw.cutoff

    def initiate_discard(self, qubit: MemoryQubit, fib_entry: FibEntry, *, round=-1):
        """
        Discard a qubit that has exceeded cutoff time at the local forwarder.

        This is called by CutoffScheme subclass.

        Args:
            round: -1 for swap_cutoff; nonnegative number for purif round.
        """
        fw = self.fw

        # Find EPR partner.
        assert qubit.partner
        partner, p_key = qubit.partner
        log.debug(
            f"{self.fw}: local cutoff discard key={qubit.key} addr={qubit.addr} round={round} partner={partner.name}:{p_key}"
        )

        # Discard primary qubit.
        fw.cnt.increment_n_cutoff(round, True)
        fw.release_qubit(qubit, need_remove=True)

        # Ask partner to discard secondary qubit.
        msg: CutoffDiscardMsg = {
            "cmd": "CUTOFF_DISCARD",
            "path_id": fib_entry.path_id,
            "key": p_key,
            "round": round,
        }
        fw.send_msg(partner, msg, fib_entry)

    def handle_discard(self, msg: CutoffDiscardMsg):
        """
        Discard a qubit that has exceeded cutoff time at the remote forwarder.

        This is called by ProactiveForwarder upon receiving a CUTOFF_DISCARD message.
        """
        fw = self.fw
        o_key = msg["key"]
        round = msg["round"]

        # Find qubit.
        qm_tuple = fw.memory.read(o_key, remove=True)
        if qm_tuple is None:
            log.debug(f"{self.fw}: remote cutoff discard key={o_key} not exist")
            return
        qubit, _ = qm_tuple
        log.debug(f"{self.fw}: remote cutoff discard key={o_key} addr={qubit.addr} round={round}")

        # Discard secondary qubit.
        fw.cnt.increment_n_cutoff(round, False)
        fw.release_qubit(qubit)

    @abstractmethod
    def before_store_eligible(self, mq: MemoryQubit, dir: PathDirection, fib_entry: FibEntry | None) -> None:
        """
        Handle an ELIGIBLE qubit stored for future swapping.
        """

    @abstractmethod
    def before_swap(self, mq0: MemoryQubit, mq1: MemoryQubit, fib_entry: FibEntry | None) -> None:
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
        fib_entry: FibEntry,
        *,
        t: Time,
        round: int,
        eligible_t: Time,
    ):
        super().__init__(t, f"addr={qubit.addr} key={qubit.key}")
        self.cutoff = cutoff
        self.qubit = qubit
        self.fib_entry = fib_entry
        self.round = round
        self.eligible_t = eligible_t

    @override
    def invoke(self) -> None:
        self.cutoff.initiate_discard(self.qubit, self.fib_entry, round=self.round)


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

    _DIR_OFFSET = {
        PathDirection.L: -2,
        PathDirection.R: -1,
    }

    def __init__(self, name="wait-time"):
        super().__init__(name)

        self.cnt = CutoffSchemeWaitTimeCounters()

    @override
    def before_store_eligible(self, mq: MemoryQubit, dir: PathDirection, fib_entry: FibEntry | None) -> None:
        if not fib_entry or (wait_budget := fib_entry.swap_cutoff[2 * fib_entry.own_idx + self._DIR_OFFSET[dir]]) is None:
            return

        now = self.simulator.tc
        deadline = now + wait_budget
        self.simulator.add_event(event := CutoffDiscardEvent(self, mq, fib_entry, round=-1, eligible_t=now, t=deadline))
        mq.events.add(event)

    @override
    def before_swap(self, mq0: MemoryQubit, mq1: MemoryQubit, fib_entry: FibEntry | None) -> None:
        _ = fib_entry
        assert mq0.events.get(CutoffDiscardEvent) is None

        event = mq1.events.discard(CutoffDiscardEvent)
        if event and self.cnt.wait_values is not None:
            self.cnt.wait_values.append((self.simulator.tc - event.eligible_t).time_slot)
