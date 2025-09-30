from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mqns.entity.memory import MemoryQubit, PathDirection, QuantumMemory
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import WernerStateEntanglement
from mqns.network.proactive.fib import Fib, FibEntry
from mqns.network.proactive.message import PathInstructions

if TYPE_CHECKING:
    from mqns.network.proactive.forwarder import ProactiveForwarder


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
    def fib(self) -> Fib:
        return self.fw.fib

    @abstractmethod
    def validate_path_instructions(self, instructions: PathInstructions) -> None:
        """Validate install_path instructions are compatible."""
        pass

    @abstractmethod
    def install_path_neighbor(
        self,
        instructions: PathInstructions,
        fib_entry: FibEntry,
        direction: PathDirection,
        neighbor: QNode,
        qchannel: QuantumChannel,
    ) -> None:
        """
        Store information about neighbor node and allocate resources.

        Args:
            instructions: Path instructions.
            fib_entry: FIB entry derived from path instructions.
            direction: LEFT for left neighbor or RIGHT for right neighbor.
            neighbor: Neighbor node.
            qchannel: Quantum channel to the neighbor.
        """
        pass

    @abstractmethod
    def uninstall_path_neighbor(
        self,
        fib_entry: FibEntry,
        direction: PathDirection,
        neighbor: QNode,
        qchannel: QuantumChannel,
    ) -> None:
        """
        Erase information about neighbor node and deallocate resources.

        Args:
            fib_entry: FIB entry.
            direction: LEFT for left neighbor or RIGHT for right neighbor.
            neighbor: Neighbor node.
            qchannel: Quantum channel to the neighbor.
        """
        pass

    @abstractmethod
    def qubit_has_path_id(self) -> bool:
        """
        Indicate whether each memory qubit shall be assigned to specific path_id.
        """
        pass

    @abstractmethod
    def qubit_is_entangled(self, qubit: MemoryQubit, neighbor: QNode) -> None:
        """
        Handle a qubit entering ENTANGLED state, i.e. having an elementary entanglement.

        This can only be invoked in ASYNC timing mode or INTERNAL phase.
        """
        pass

    @abstractmethod
    def find_swap_candidate(
        self, qubit: MemoryQubit, epr: WernerStateEntanglement, fib_entry: FibEntry | None
    ) -> tuple[MemoryQubit, FibEntry] | None:
        """
        Find another qubit to swap with an ELIGIBLE qubit.

        Args:
            qubit: A qubit in ELIGIBLE state.
            epr: The EPR associated with this qubit. This is not an end-to-end entanglement.
            fib_entry: FIB entry passed to `fw.qubit_is_eligible()`.

        Returns:
            None: No candidate, do not swap.
            [0]: Another qubit in ELIGIBLE state.
            [1]: FIB entry for `fw.do_swapping()`.
        """
        pass

    @abstractmethod
    def swapping_succeeded(
        self,
        prev_epr: WernerStateEntanglement,
        next_epr: WernerStateEntanglement,
        new_epr: WernerStateEntanglement,
    ) -> None:
        """
        Handle a successful swap at the swapping node.

        Args:
            prev_epr: An EPR with a partner node to the left.
            next_epr: An EPR with a partner node to the right
            new_epr: Locally swapped EPR made from prev_epr+next_epr.
        """
        pass

    @abstractmethod
    def su_parallel_has_conflict(self, my_new_epr: WernerStateEntanglement, su_path_id: int) -> bool:
        """
        Determine whether a parallel SWAP_UPDATE has a conflict.

        Args:
            my_new_epr: Locally swapped EPR.
            su_path_id: The path_id chosen by another node performing paralleel swapping.

        Returns:
            If True, a conflict is detected and the SWAP_UPDATE is discarded.
            Otherwise, the SWAP_UPDATE continues processing.
        """
        pass

    @abstractmethod
    def su_parallel_succeeded(
        self, merged_epr: WernerStateEntanglement, new_epr: WernerStateEntanglement, other_epr: WernerStateEntanglement
    ) -> None:
        """
        Handle a successful parallel swap at the recipient of SWAP_UPDATE message.

        See the diagram in `ProactiveForwarder._su_parallel` for an explanation of the arguments.

        Args:
            merged_epr: Locally merged EPR made from other_epr+new_epr.
            new_epr: Remotely swapped EPR from the sender of SWAP_UPDATE message.
            other_epr: An EPR between local and the other partner.
        """
        pass
