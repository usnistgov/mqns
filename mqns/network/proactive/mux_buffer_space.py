import random
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, override

from mqns.entity.memory import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.proactive.fib import FibEntry
from mqns.network.proactive.message import PathInstructions, validate_path_instructions
from mqns.network.proactive.mux import MuxScheme
from mqns.network.proactive.select import MemoryEprIterator, MemoryEprTuple
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.proactive.forwarder import ProactiveForwarder


class MuxSchemeFibBase(MuxScheme):
    SelectSwapQubit = Callable[["ProactiveForwarder", MemoryEprTuple, FibEntry, list[MemoryEprTuple]], MemoryEprTuple]

    SelectSwapQubit_random: SelectSwapQubit = lambda _fw, _mt, _fe, candidates: random.choice(candidates)

    def __init__(self, name: str, select_swap_qubit: SelectSwapQubit | None):
        super().__init__(name)
        self._select_swap_qubit = select_swap_qubit

    @override
    def find_swap_candidate(
        self, qubit: MemoryQubit, epr: Entanglement, fib_entry: FibEntry | None, input: MemoryEprIterator
    ) -> tuple[MemoryQubit, FibEntry] | None:
        _ = epr
        assert fib_entry is not None

        candidates = self.list_swap_candidates(qubit, fib_entry, input)
        if self._select_swap_qubit is None:
            mt1 = next(candidates, None)
            return None if mt1 is None else (mt1[0], fib_entry)

        candidates = list(candidates)
        if len(candidates) == 0:
            return None
        mt1 = self._select_swap_qubit(self.fw, (qubit, epr), fib_entry, candidates)
        return mt1[0], fib_entry

    @abstractmethod
    def list_swap_candidates(self, mq0: MemoryQubit, fib_entry: FibEntry, input: MemoryEprIterator) -> MemoryEprIterator:
        pass


class MuxSchemeBufferSpace(MuxSchemeFibBase):
    """
    Buffer-Space multiplexing scheme.
    """

    def __init__(
        self,
        name="buffer-space multiplexing",
        *,
        select_swap_qubit: MuxSchemeFibBase.SelectSwapQubit | None = None,
    ):
        """
        Args:
            select_swap_qubit: Function to select a qubit to swap with, default is first.
        """
        super().__init__(name, select_swap_qubit)

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
        m_v_offset, ch_side = (-1, 1) if direction == PathDirection.L else (0, 0)

        n_qubits = m_v[fib_entry.own_idx + m_v_offset][ch_side]
        addrs = self.memory.allocate(
            qchannel,
            fib_entry.path_id,
            direction,
            n="all" if n_qubits == 0 else n_qubits,
        )
        log.debug(f"{self.node}: allocated {direction} qubits: {addrs}")

    @override
    def uninstall_path_neighbor(
        self, fib_entry: FibEntry, direction: PathDirection, neighbor: QNode, qchannel: QuantumChannel
    ) -> None:
        _ = neighbor
        qubits = self.memory.find(lambda q, _: q.path_id == fib_entry.path_id, qchannel=qchannel)
        addrs = [q[0].addr for q in qubits]
        self.memory.deallocate(*addrs)
        log.debug(f"{self.node}: deallocated {direction} qubits: {addrs}")
        pass

    @override
    def qubit_has_path_id(self) -> bool:
        return True

    @override
    def qubit_is_entangled(self, qubit: MemoryQubit, epr: Entanglement, neighbor: QNode) -> None:
        _ = epr
        if qubit.path_id is None:
            log.debug(f"{self.node}: release entangled qubit {qubit.addr} due to uninstalled path")
            self.fw.release_qubit(qubit, need_remove=True)
            return

        fib_entry = self.fib.get(qubit.path_id)
        qubit.purif_rounds = 0
        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, neighbor)

    @override
    def list_swap_candidates(self, mq0: MemoryQubit, fib_entry: FibEntry, input: MemoryEprIterator):
        assert mq0.path_id is not None
        possible_path_ids = {fib_entry.path_id}

        return (
            (q, v)
            for (q, v) in input
            if q.path_id in possible_path_ids  # allocated to the same path_id or another path_id under the same request_id
            and q.path_direction != mq0.path_direction  # in the opposite path direction
        )

    @override
    def swapping_succeeded(self, prev_epr: Entanglement, next_epr: Entanglement, new_epr: Entanglement) -> None:
        assert prev_epr.tmp_path_ids is None
        assert next_epr.tmp_path_ids is None
        _ = new_epr

    @override
    def su_parallel_has_conflict(self, my_new_epr: Entanglement, su_path_id: int) -> bool:
        assert my_new_epr.tmp_path_ids is None
        _ = su_path_id
        return False

    @override
    def su_parallel_succeeded(self, merged_epr: Entanglement, new_epr: Entanglement, other_epr: Entanglement) -> None:
        assert new_epr.tmp_path_ids is None
        assert other_epr.tmp_path_ids is None
        _ = merged_epr
