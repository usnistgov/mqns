from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from qns.entity.memory import MemoryQubit, QuantumMemory
from qns.entity.node import QNode
from qns.models.epr import WernerStateEntanglement
from qns.network.proactive.fib import FIBEntry, ForwardingInformationBase
from qns.network.proactive.message import InstallPathInstructions

if TYPE_CHECKING:
    from qns.network.proactive.forwarder import ProactiveForwarder


class MuxScheme(ABC):
    """Path multiplexing scheme."""

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
    def memory(self) -> QuantumMemory:
        return self.fw.memory

    @property
    def fib(self) -> ForwardingInformationBase:
        return self.fw.fib

    @abstractmethod
    def validate_path_instructions(self, instructions: InstallPathInstructions) -> None:
        """Validate install_path instructions are compatible."""
        pass

    @abstractmethod
    def qubit_is_entangled(self, qubit: MemoryQubit, neighbor: QNode) -> None:
        pass

    @abstractmethod
    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FIBEntry | None) -> None:
        pass

    @abstractmethod
    def swapping_succeeded(
        self,
        prev_epr: WernerStateEntanglement,
        next_epr: WernerStateEntanglement,
        new_epr: WernerStateEntanglement,
    ) -> None:
        pass

    @abstractmethod
    def su_parallel_avoid_conflict(self, my_new_epr: WernerStateEntanglement, su_path_id: int) -> bool:
        pass

    @abstractmethod
    def su_parallel_succeeded(
        self, merged_epr: WernerStateEntanglement, new_epr: WernerStateEntanglement, other_epr: WernerStateEntanglement
    ) -> None:
        pass
