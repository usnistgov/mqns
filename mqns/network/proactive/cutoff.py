from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from typing_extensions import override

from mqns.entity.memory import MemoryQubit
from mqns.entity.node import QNode
from mqns.models.epr import WernerStateEntanglement
from mqns.network.proactive.fib import FibEntry
from mqns.network.proactive.message import CutoffDiscardMsg
from mqns.simulator import Simulator, func_to_event
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.proactive.forwarder import ProactiveForwarder


class CutoffScheme(ABC):
    """
    EPR age cut-off scheme.

    This determines how PathInstructions.swap_cutoff is interpreted.
    """

    def __init__(self, name: str):
        self.name = name
        """Scheme name."""

        self.fw: "ProactiveForwarder"
        """
        Forwarder that uses this instance, assigned by the forwarder install function.
        """

    def __repr__(self):
        return f"<{self.name}>"

    @property
    def own(self) -> QNode:
        return self.fw.own

    @property
    def simulator(self) -> Simulator:
        return self.own.simulator

    def initiate_discard(self, qubit: MemoryQubit, fib_entry: FibEntry, *, round=-1):
        """
        Discard a qubit that has exceeded cutoff time at the local forwarder.

        This is called by CutoffScheme subclass.

        Args:
            round: -1 for swap_cutoff; nonnegative number for purif round.
        """
        fw = self.fw

        # find EPR partner
        _, epr = fw.memory.read(qubit.addr, must=True)
        assert isinstance(epr, WernerStateEntanglement)
        partner = epr.dst if epr.src == self.own else epr.src
        assert partner is not None

        log.debug(f"{self.own}: local cutoff discard epr={epr.name} addr={qubit.addr} round={round} partner={partner.name}")

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
        qm_tuple = fw.memory.read(epr_name)
        if qm_tuple is None:
            log.debug(f"{self.own}: remote cutoff discard epr={epr_name} not exist")
            return
        qubit, _ = qm_tuple
        log.debug(f"{self.own}: remote cutoff discard epr={epr_name} addr={qubit.addr} round={round}")

        # discard secondary qubit
        fw.cnt.increment_n_cutoff(round, False)
        fw.release_qubit(qubit)

    @abstractmethod
    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FibEntry | None) -> None:
        """
        Handle a qubit that has become ELIGIBLE for swapping.
        The qubit is not to be consumed.
        """
        pass

    @abstractmethod
    def filter_swap_candidate(self, qubit: MemoryQubit) -> bool:
        """
        Determine whether a qubit can be used as swap candidate.
        """
        pass

    @abstractmethod
    def take_qubit(self, qubit: MemoryQubit) -> None:
        """
        Mark a qubit as taken/used in purification or swapping.
        """
        pass


class CutoffSchemeWaitTime(CutoffScheme):
    """
    EPR age cut-off with individual wait-time budget.
    """

    def __init__(self, name="wait-time"):
        super().__init__(name)

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

        discard_event = func_to_event(deadline, self.initiate_discard, qubit, fib_entry)
        qubit.set_event(CutoffSchemeWaitTime, discard_event)
        self.simulator.add_event(discard_event)

    @override
    def filter_swap_candidate(self, qubit: MemoryQubit) -> bool:
        return qubit.cutoff is None or qubit.cutoff[1] >= self.simulator.tc

    @override
    def take_qubit(self, qubit: MemoryQubit) -> None:
        if qubit.cutoff is None:
            return
        qubit.set_event(CutoffSchemeWaitTime, None)
        qubit.cutoff = None


class CutoffSchemeWernerAge(CutoffScheme):
    """
    EPR age cut-off with accumulated Werner age metric.
    """

    def __init__(self, name="wait-time"):
        super().__init__(name)
