from abc import abstractmethod

from typing_extensions import override

from mqns.entity.memory import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import WernerStateEntanglement
from mqns.network.proactive.fib import FibEntry
from mqns.network.proactive.message import PathInstructions, validate_path_instructions
from mqns.network.proactive.mux import MuxScheme
from mqns.network.proactive.select import MemoryWernerIterator, select_swap_qubit
from mqns.utils import log


class MuxSchemeFibBase(MuxScheme):
    @override
    def find_swap_candidate(
        self, qubit: MemoryQubit, epr: WernerStateEntanglement, fib_entry: FibEntry | None, input: MemoryWernerIterator
    ) -> tuple[MemoryQubit, FibEntry] | None:
        _ = epr
        assert fib_entry is not None
        found = select_swap_qubit(
            self.fw._select_swap_qubit, qubit, epr, fib_entry, self.list_swap_candidates(qubit, fib_entry, input)
        )
        if not found:
            return None
        return found[0], fib_entry

    @abstractmethod
    def list_swap_candidates(self, mq0: MemoryQubit, fib_entry: FibEntry, input: MemoryWernerIterator) -> MemoryWernerIterator:
        pass


class MuxSchemeBufferSpace(MuxSchemeFibBase):
    """
    Buffer-Space multiplexing scheme.
    """

    def __init__(self, name="buffer-space multiplexing"):
        super().__init__(name)

    @override
    def validate_path_instructions(self, instructions: PathInstructions) -> None:
        validate_path_instructions(instructions)
        assert "m_v" in instructions

    @override
    def install_path_neighbor(
        self,
        instructions: PathInstructions,
        fib_entry: FibEntry,
        direction: PathDirection,
        neighbor: QNode,
        qchannel: QuantumChannel,
    ) -> None:
        _ = neighbor
        assert "m_v" in instructions
        m_v = instructions["m_v"]
        m_v_offset, ch_side = (-1, 1) if direction == PathDirection.LEFT else (0, 0)

        n_qubits = m_v[fib_entry.own_idx + m_v_offset][ch_side]
        if n_qubits == 0:  # 0 means use all qubits assigned to this qchannel
            n_qubits = len(self.memory.get_channel_qubits(qchannel))

        addrs = self.memory.allocate(qchannel, fib_entry.path_id, direction, n=n_qubits)
        log.debug(f"{self.own}: allocated {direction} qubits: {addrs}")

    @override
    def uninstall_path_neighbor(
        self,
        fib_entry: FibEntry,
        direction: PathDirection,
        neighbor: QNode,
        qchannel: QuantumChannel,
    ) -> None:
        _ = neighbor
        qubits = self.memory.find(lambda q, _: q.path_id == fib_entry.path_id, qchannel=qchannel)
        addrs = [q[0].addr for q in qubits]
        self.memory.deallocate(*addrs)
        log.debug(f"{self.own}: deallocated {direction} qubits: {addrs}")
        pass

    @override
    def qubit_has_path_id(self) -> bool:
        return True

    @override
    def qubit_is_entangled(self, qubit: MemoryQubit, neighbor: QNode) -> None:
        if qubit.path_id is None:
            log.debug(f"{self.own}: release entangled qubit {qubit.addr} due to uninstalled path")
            self.fw.release_qubit(qubit, need_remove=True)
            return

        fib_entry = self.fib.get(qubit.path_id)
        qubit.purif_rounds = 0
        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, neighbor)

    @override
    def list_swap_candidates(self, mq0: MemoryQubit, fib_entry: FibEntry, input: MemoryWernerIterator):
        assert mq0.path_id is not None
        possible_path_ids = {fib_entry.path_id}

        return (
            (q, v)
            for (q, v) in input
            if q.path_id in possible_path_ids  # allocated to the same path_id or another path_id under the same request_id
            and q.path_direction != mq0.path_direction  # in the opposite path direction
        )

    @override
    def swapping_succeeded(
        self,
        prev_epr: WernerStateEntanglement,
        next_epr: WernerStateEntanglement,
        new_epr: WernerStateEntanglement,
    ) -> None:
        assert prev_epr.tmp_path_ids is None
        assert next_epr.tmp_path_ids is None
        _ = new_epr

    @override
    def su_parallel_has_conflict(self, my_new_epr: WernerStateEntanglement, su_path_id: int) -> bool:
        assert my_new_epr.tmp_path_ids is None
        _ = su_path_id
        return False

    @override
    def su_parallel_succeeded(
        self, merged_epr: WernerStateEntanglement, new_epr: WernerStateEntanglement, other_epr: WernerStateEntanglement
    ) -> None:
        assert new_epr.tmp_path_ids is None
        assert other_epr.tmp_path_ids is None
        _ = merged_epr
