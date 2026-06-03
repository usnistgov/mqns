from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, cast, override

from mqns.entity.memory import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.fw.fib import FibEntry
from mqns.network.fw.message import PathInstructions, validate_path_instructions
from mqns.network.fw.mux import MuxScheme
from mqns.network.fw.select import MemoryEprIterator, MemoryEprTuple, call_select, select_random

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class MuxSchemeDynamicBase(MuxScheme):
    def __init__(self, name: str):
        super().__init__(name)
        self.qchannel_paths_map = defaultdict[str, list[int]](lambda: [])
        """stores path-qchannel relationship"""

    @override
    def validate_path_instructions(self, instructions: PathInstructions):
        validate_path_instructions(instructions)
        assert "m_v" not in instructions

    @override
    def install_path_neighbor(
        self,
        instructions: PathInstructions,
        fib_entry: FibEntry,
        direction: PathDirection,
        neighbor: QNode,
        qchannel: QuantumChannel,
    ) -> None:
        _ = instructions
        _ = direction
        _ = neighbor
        self.qchannel_paths_map[qchannel.name].append(fib_entry.path_id)

    @override
    def uninstall_path_neighbor(
        self,
        fib_entry: FibEntry,
        direction: PathDirection,
        neighbor: QNode,
        qchannel: QuantumChannel,
    ) -> None:
        _ = direction
        _ = neighbor
        paths = self.qchannel_paths_map[qchannel.name]
        paths.remove(fib_entry.path_id)
        if len(paths) == 0:
            del self.qchannel_paths_map[qchannel.name]

    @override
    def qubit_has_path_id(self) -> bool:
        return False

    @override
    def list_qubit_epr_path_ids(self, mq: MemoryQubit) -> list[int]:
        assert mq.path_id is None
        assert mq.qchannel, f"{self.fw}: No qubit-qchannel assignment. Not supported."
        return self.qchannel_paths_map.get(mq.qchannel.name, [])


class MuxSchemeStatistical(MuxSchemeDynamicBase):
    """
    Statistical multiplexing scheme.
    """

    type SelectSwapQubit = Callable[[list[MemoryEprTuple], "Forwarder", MemoryEprTuple], MemoryEprTuple]

    SelectSwapQubit_random: SelectSwapQubit = select_random

    type SelectPath = Callable[[list[int], "Forwarder", Entanglement, Entanglement], int | FibEntry]

    SelectPath_random: SelectPath = select_random

    def __init__(
        self,
        name="statistical multiplexing",
        *,
        select_swap_qubit: SelectSwapQubit | None = None,
        coordinated_decisions=False,
        select_path: SelectPath | None = None,
    ):
        """
        Args:
            select_swap_qubit: Function to select a qubit to swap with, default is first.
            coordinated_decisions:
                If True, during a parallel swap, the path_id chosen at one node for selecting swap candidates
                is instantly visible at other nodes. This behavior is physically unrealistic. It is implemented
                for comparison purpose.
                If False (default), during a parallel swap, each node selects swap candidates independently,
                and then discards unusable entanglements due to conflictual swap decisions.
            select_path: Function to select a path (FIB entry) to swap into, default is first.
                This has no effect unless coordinated_decisions is True.
        """
        super().__init__(name)
        self._select_swap_qubit = select_swap_qubit
        self.coordinated_decisions = coordinated_decisions
        self._select_path = select_path

    @override
    def validate_path_instructions(self, instructions: PathInstructions):
        super().validate_path_instructions(instructions)

        # swap sequence must be [1, 0, 0, .., 0, 0, 1]
        s0, *s1, s2 = instructions["swap"]
        assert s0 == 1 == s2
        assert all((s == 0 for s in s1))

        # purif scheme must be empty / zeros
        assert all((r == 0 for r in instructions["purif"].values()))

    @override
    def qubit_is_entangled(self, mq: MemoryQubit, epr: Entanglement, neighbor: QNode) -> FibEntry | None:
        if self.coordinated_decisions and epr.affectionated_path_id >= 0:
            assert epr.affectionated_path_id in cast(list[int], mq.epr_path_ids)
            mq.epr_path_ids = [epr.affectionated_path_id]

        def calc_rank_diff(path_id: int):
            fib_entry = self.fib.get(path_id)
            _, p_rank = fib_entry.find_index_and_swap_rank(neighbor.name)
            return fib_entry.own_swap_rank - p_rank

        rank_diff = [calc_rank_diff(path_id) for path_id in cast(list[int], mq.epr_path_ids)]
        assert min(rank_diff) == max(rank_diff)  # failure means one route is a substring of another route, unsupported
        if rank_diff[0] > 0:
            # Own node has higher rank and cannot initiate swap; qubit stays in ENTANGLED1 state.
            return None

        # Own node has lower/equal rank and can initiate swap.
        mq.state = QubitState.PURIF
        # Without FIB entry, purification scheme cannot be specified, set ELIGIBLE for swapping right away.
        mq.state = QubitState.ELIGIBLE
        return None

    @override
    def find_swap_candidate(
        self, mq0: MemoryQubit, epr0: Entanglement, fib_entry: FibEntry | None, input: MemoryEprIterator
    ) -> tuple[MemoryQubit, FibEntry] | None:
        mq0_path_ids = set(cast(list[int], mq0.epr_path_ids))

        # find another qubit to swap with
        mt1 = call_select(
            (
                (q, v)
                for (q, v) in input
                if not mq0_path_ids.isdisjoint(cast(list[int], q.epr_path_ids))  # has overlapping epr_path_ids
            ),
            self._select_swap_qubit,
            self.fw,
            (mq0, epr0),
        )
        if mt1 is None:
            return None
        mq1, epr1 = mt1
        assert type(epr1) is self.fw.epr_type

        # select a FIB entry to guide swap updates
        selected_path = call_select(
            sorted(mq0_path_ids.intersection(cast(list[int], mq1.epr_path_ids))),
            self._select_path,
            self.fw,
            epr0,
            epr1,
        )
        assert selected_path is not None
        fib_entry = selected_path if type(selected_path) is FibEntry else self.fib.get(selected_path)
        if self.coordinated_decisions:
            epr0.affectionated_path_id = epr1.affectionated_path_id = fib_entry.path_id
        return mq1, fib_entry
