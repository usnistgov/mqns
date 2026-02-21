from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from mqns.entity.memory import MemoryQubit
from mqns.entity.node import QNode
from mqns.network.fw import FibEntry
from mqns.network.fw.message import CutoffDiscardMsg
from mqns.simulator import Simulator, func_to_event
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.fw import Forwarder


class CutoffScheme(ABC):
    """
    EPR age cut-off scheme.

    This determines how PathInstructions.swap_cutoff is interpreted.
    """

    def __init__(self, name: str):
        self.name = name
        """Scheme name."""

        self.fw: "Forwarder"
        """
        Forwarder that uses this instance, assigned by the forwarder install function.
        """

    def __repr__(self):
        return f"<{self.name}>"

    @classmethod
    def of(cls, fw: "Forwarder"):
        """
        Retrieve subclass instance from forwarder.
        """
        assert isinstance(fw.cutoff, cls)
        return fw.cutoff

    @property
    def node(self) -> QNode:
        return self.fw.node

    @property
    def simulator(self) -> Simulator:
        return self.node.simulator

    def initiate_discard(self, qubit: MemoryQubit, fib_entry: FibEntry, *, round=-1):
        """
        Discard a qubit that has exceeded cutoff time at the local forwarder.

        This is called by CutoffScheme subclass.

        Args:
            round: -1 for swap_cutoff; nonnegative number for purif round.
        """
        fw = self.fw

        # find EPR partner
        _, epr = fw.memory.read(qubit.addr, has=self.fw.epr_type, set_fidelity=True, remove=True)
        partner = epr.dst if epr.src == self.node else epr.src
        assert partner is not None

        log.debug(f"{self.node}: local cutoff discard epr={epr.name} addr={qubit.addr} round={round} partner={partner.name}")

        # discard primary qubit
        fw.cnt.increment_n_cutoff(round, True)
        fw.release_qubit(qubit)

        # ask partner to discard secondary qubit
        msg: CutoffDiscardMsg = {
            "cmd": "CUTOFF_DISCARD",
            "path_id": fib_entry.path_id,
            "epr": epr.name,
            "round": round,
        }
        fw.send_msg(partner, msg, fib_entry)

    def handle_discard(self, msg: CutoffDiscardMsg):
        """
        Discard a qubit that has exceeded cutoff time at the remote forwarder.

        This is called by ProactiveForwarder upon receiving a CUTOFF_DISCARD message.
        """
        fw = self.fw
        epr_name = msg["epr"]
        round = msg["round"]

        # find qubit
        qm_tuple = fw.memory.read(epr_name, remove=True)
        if qm_tuple is None:
            log.debug(f"{self.node}: remote cutoff discard epr={epr_name} not exist")
            return
        qubit, _ = qm_tuple
        log.debug(f"{self.node}: remote cutoff discard epr={epr_name} addr={qubit.addr} round={round}")

        # discard secondary qubit
        fw.cnt.increment_n_cutoff(round, False)
        fw.release_qubit(qubit)

    @abstractmethod
    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FibEntry | None) -> None:
        """
        Handle a qubit that has become ELIGIBLE for swapping.
        The qubit would not be consumed.
        """
        pass

    @abstractmethod
    def filter_swap_candidate(self, qubit: MemoryQubit) -> bool:
        """
        Determine whether a qubit can be used as swap candidate.
        """
        pass

    @abstractmethod
    def before_swap(self, mq0: MemoryQubit, mq1: MemoryQubit | None, fib_entry: FibEntry | None) -> None:
        """
        Handle a pair of qubits before swapping, or one qubits stored for future swapping.
        This should be invoked in the same time_slot as qubit_is_eligible.

        Args:
            mq0: Newly eligible qubits.
            mq1: If None, mq0 is stored; otherwise, mq0-mq1 will swap.
        """
        pass


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
    The controller provides these wait-time budgets in `PathInstructions.swap_cutoff` field.

    Each node individually tracks how long an EPR has been waiting in memory until it can be swapped.
    If an EPR has waited for more than the budget at this node, it cannot be used in a swap and should
    be released to make room for a new EPR.
    """

    def __init__(self, name="wait-time"):
        super().__init__(name)

        self.cnt = CutoffSchemeWaitTimeCounters()

    @override
    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FibEntry | None) -> None:
        qubit.cutoff = None

        if fib_entry is None:
            return
        wait_budget = fib_entry.swap_cutoff[fib_entry.own_idx]
        if wait_budget is None:
            return

        now = self.simulator.tc
        deadline = now + wait_budget
        qubit.cutoff = (now, deadline)

    @override
    def filter_swap_candidate(self, qubit: MemoryQubit) -> bool:
        return qubit.cutoff is None or qubit.cutoff[1] >= self.simulator.tc

    @override
    def before_swap(self, mq0: MemoryQubit, mq1: MemoryQubit | None, fib_entry: FibEntry | None) -> None:
        if mq0.cutoff is None:
            return
        assert fib_entry is not None

        now, deadline = mq0.cutoff
        if mq1 is None:
            discard_event = func_to_event(deadline, self.initiate_discard, mq0, fib_entry)
            mq0.set_event(CutoffSchemeWaitTime, discard_event)
            self.simulator.add_event(discard_event)
            return

        if mq1.cutoff is None:
            return

        if self.cnt.wait_values is not None:
            t0, _ = mq1.cutoff
            self.cnt.wait_values.append((now - t0).time_slot)

        mq1.set_event(CutoffSchemeWaitTime, None)
        mq1.cutoff = None
