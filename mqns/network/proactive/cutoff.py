from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from typing_extensions import override

from mqns.entity.memory import MemoryQubit
from mqns.entity.node import QNode
from mqns.models.epr import WernerStateEntanglement
from mqns.network.proactive.fib import FibEntry
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

    @abstractmethod
    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FibEntry | None) -> None:
        pass

    @abstractmethod
    def filter_swap_candidate(self, qubit: MemoryQubit) -> bool:
        pass

    @abstractmethod
    def take_qubit(self, qubit: MemoryQubit) -> None:
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

        release_event = func_to_event(deadline, self._release_qubit, qubit)
        qubit.set_event(CutoffSchemeWaitTime, release_event)
        self.simulator.add_event(release_event)

    @override
    def filter_swap_candidate(self, qubit: MemoryQubit) -> bool:
        return qubit.cutoff is None or qubit.cutoff[1] >= self.simulator.tc

    @override
    def take_qubit(self, qubit: MemoryQubit) -> None:
        if qubit.cutoff is None:
            return
        qubit.set_event(CutoffSchemeWaitTime, None)
        qubit.cutoff = None

    def _release_qubit(self, qubit: MemoryQubit) -> None:
        _, epr = self.fw.memory.read(qubit.addr, must=True)
        assert isinstance(epr, WernerStateEntanglement)

        partner = epr.dst if epr.src == self.own else epr.src
        assert partner is not None
        assert partner.memory is not None
        partner_qm = partner.memory.read(epr)

        self.fw.release_qubit(qubit)
        self.fw.cnt.n_swap_cutoff[0] += 1
        log.debug(f"{self.own}: qubit wait-time exceeded addr={qubit.addr} secondary-partner={partner.name}")
        if partner_qm is not None:
            partner_qubit, _ = partner_qm
            partner_fw = partner.get_app(type(self.fw))
            partner_fw.cnt.n_swap_cutoff[1] += 1
            partner_fw.release_qubit(partner_qubit)
            log.debug(f"{partner}: qubit wait-time exceeded addr={qubit.addr} primary-partner={self.own.name}")
        # TODO release qubit at partner_fw via message instead of function call


class CutoffSchemeWernerAge(CutoffScheme):
    """
    EPR age cut-off with accumulated Werner age metric.
    """

    def __init__(self, name="wait-time"):
        super().__init__(name)
