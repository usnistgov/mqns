from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from typing_extensions import override

from mqns.entity.memory import MemoryQubit
from mqns.entity.node import QNode
from mqns.network.proactive.fib import FibEntry
from mqns.simulator import Simulator

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

    @abstractmethod
    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FibEntry | None) -> None:
        pass

    @abstractmethod
    def filter_swap_candidate(self, qubit: MemoryQubit) -> bool:
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
        qubit.cutoff = (now, now + wait_budget)

    @override
    def filter_swap_candidate(self, qubit: MemoryQubit) -> bool:
        return qubit.cutoff is None or qubit.cutoff[1] >= self.simulator.tc


class CutoffSchemeWernerAge(CutoffScheme):
    """
    EPR age cut-off with accumulated Werner age metric.
    """

    def __init__(self, name="wait-time"):
        super().__init__(name)
