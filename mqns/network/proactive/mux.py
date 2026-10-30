from abc import ABC, abstractmethod

from mqns.entity.memory import MemoryQubit, PathDirection, QubitState
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.fw import FibPath, ForwarderModule, PathInstructions


class MuxScheme(ForwarderModule, ABC):
    """Path multiplexing scheme."""

    @abstractmethod
    def validate_path_instructions(self, inst: PathInstructions) -> None:
        """Validate path instructions are compatible."""

    @abstractmethod
    def install_path_adj(self, inst: PathInstructions, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        """
        Store information about adjacent quantum channel, allocate resources, activate LinkLayer.

        Args:
            instructions: Path instructions.
            fp: FIB path entry.
            dir: Direction of adjacency.
            ch: Quantum channel to the neighbor.
        """

    @abstractmethod
    def uninstall_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        """
        Erase information about adjacent quantum channel, deallocate resources, deactivate LinkLayer.

        Args:
            fp: FIB path entry.
            dir: Direction of adjacency.
            ch: Quantum channel to the neighbor.
        """

    @abstractmethod
    def list_qubit_epr_path_ids(self, mq: MemoryQubit) -> list[int]:
        """
        Compute ``mq.epr_path_ids`` on a qubit entering ENTANGLED1 state.
        """

    @abstractmethod
    def qubit_is_entangled(self, mq: MemoryQubit, epr: Entanglement) -> FibPath | None:
        """
        Handle a qubit entering ENTANGLED state, i.e. having an elementary entanglement.

        Pre-conditions:

        * The network is in ASYNC timing mode or INTERNAL phase.
        * ``mq.partner`` is assigned.
        * ``mq.epr_path_ids`` is populated and non-empty.
        * ``epr`` is an elementary entanglement.

        Post-condition and return value:

        * ``mq.state is PURIF``: forwarder starts purification; FIB entry is required.
        * ``mq.state is ELIGIBLE``: forwarder starts swapping; FIB entry is optional.
        """

    @abstractmethod
    def find_swap_with(self, mq0: MemoryQubit, epr0: Entanglement, fp: FibPath | None) -> tuple[MemoryQubit, FibPath] | None:
        """
        Choose another qubit to swap with a qubit entering ELIGIBLE state and ready to swap.

        Args:
            mq0: The qubit entering ELIGIBLE state.
            epr0: The entanglement stored in the qubit.
            fp: FIB path entry found by ``MuxScheme.qubit_is_entangled`` or used in last round of purification.

        Returns:
            None: Do not swap.
            [0]: The other qubit, which must be in ELIGIBLE state.
            [1]: FIB entry to guide ``ForwarderSwapProc``.
        """

    @staticmethod
    def qubits_swappable(mq0: MemoryQubit, mq1: MemoryQubit) -> bool:
        """
        Determine whether ``mq1`` in memory can swap with ``mq0``.
        """
        return mq1.state is QubitState.ELIGIBLE and mq1.qchannel is not mq0.qchannel
