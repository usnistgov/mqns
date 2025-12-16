import random
from collections import defaultdict
from collections.abc import Iterable, Set

from typing_extensions import override

from mqns.entity.memory import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import BaseEntanglement, WernerStateEntanglement
from mqns.network.proactive.fib import FibEntry
from mqns.network.proactive.message import PathInstructions, validate_path_instructions
from mqns.network.proactive.mux import MuxScheme
from mqns.network.proactive.select import MemoryWernerIterator, select_swap_qubit
from mqns.utils import log


def has_intersect_tmp_path_ids(epr0: Set[int] | None, epr1: Iterable[int] | None) -> bool:
    """
    Determine whether at least one path_id overlaps between tmp_path_ids sets in two EPRs.
    """
    return epr0 is not None and epr1 is not None and not epr0.isdisjoint(epr1)


def intersect_tmp_path_ids(epr0: BaseEntanglement, epr1: BaseEntanglement) -> frozenset[int]:
    """
    Find overlapping path_ids between tmp_path_ids sets in two EPRs.
    """
    assert epr0.tmp_path_ids is not None
    assert epr1.tmp_path_ids is not None
    path_ids = epr0.tmp_path_ids.intersection(epr1.tmp_path_ids)
    if not path_ids:
        raise Exception(f"Cannot select path ID from {epr0.tmp_path_ids} and {epr1.tmp_path_ids}")
    return path_ids


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

    def _qubit_is_entangled_0(self, qubit: MemoryQubit) -> list[int]:
        assert qubit.path_id is None
        assert qubit.qchannel is not None, f"{self.own}: No qubit-qchannel assignment. Not supported."

        possible_path_ids = self.qchannel_paths_map.get(qubit.qchannel.name, [])
        if not possible_path_ids:
            log.debug(f"{self.own}: release entangled qubit {qubit.addr} due to uninstalled path")
            self.fw.release_qubit(qubit, need_remove=True)

        return possible_path_ids


class MuxSchemeStatistical(MuxSchemeDynamicBase):
    """
    Statistical multiplexing scheme.
    """

    def __init__(
        self,
        name="statistical multiplexing",
        *,
        coordinated_decisions=False,
    ):
        """
        Args:
            coordinated_decisions:
                If True, during a parallel swap, the path_id chosen at one node for selecting swap candidates
                is instantly visible at other nodes. This behavior is physically unrealistic. It is implemented
                for comparison purpose.
                If False (default), during a parallel swap, each node selects swap candidates independently,
                and then discards unusable entanglements due to conflictual swap decisions.
        """
        super().__init__(name)
        self.coordinated_decisions = coordinated_decisions

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
    def qubit_is_entangled(self, qubit: MemoryQubit, neighbor: QNode) -> None:
        _ = neighbor
        possible_path_ids = frozenset(self._qubit_is_entangled_0(qubit))
        if not possible_path_ids:  # all paths on the channel have been uninstalled
            return

        _, epr = self.memory.read(qubit.addr, must=WernerStateEntanglement)

        log.debug(f"{self.own}: qubit {qubit} has tmp_path_ids {possible_path_ids}")
        if epr.tmp_path_ids is None:
            epr.tmp_path_ids = possible_path_ids
        elif self.coordinated_decisions:
            assert epr.tmp_path_ids.issubset(possible_path_ids)
        else:
            # Assuming both primary and secondary nodes in an elementary EPR have the same path instructions,
            # both nodes should have the same qchannel_paths_map and thus derive the same tmp_path_ids.
            assert epr.tmp_path_ids == possible_path_ids

        if self._can_enter_purif(epr, neighbor):
            qubit.state = QubitState.PURIF

            # purif scheme is empty, as checked in validate_path_instructions
            log.debug(f"{self.own}: no FIB associated to qubit -> set eligible")
            qubit.state = QubitState.ELIGIBLE
            self.fw.qubit_is_eligible(qubit, None)

    def _can_enter_purif(self, epr: WernerStateEntanglement, neighbor: QNode) -> bool:
        def calc_rank_diff(path_id: int):
            fib_entry = self.fib.get(path_id)
            _, p_rank = fib_entry.find_index_and_swap_rank(neighbor.name)
            return fib_entry.own_swap_rank - p_rank

        assert epr.tmp_path_ids is not None
        rank_diff = [calc_rank_diff(path_id) for path_id in epr.tmp_path_ids]
        assert min(rank_diff) == max(rank_diff)  # failure means one route is a substring of another route, unsupported
        return rank_diff[0] <= 0

    @override
    def find_swap_candidate(
        self, qubit: MemoryQubit, epr: WernerStateEntanglement, fib_entry: FibEntry | None, input: MemoryWernerIterator
    ) -> tuple[MemoryQubit, FibEntry] | None:
        assert qubit.qchannel is not None

        # find qchannels whose qubits may be used with this qubit
        # use path_ids to look for acceptable qchannels for swapping, excluding the qubit's qchannel
        matched_channels = {
            channel
            for channel, path_ids in self.qchannel_paths_map.items()
            if channel != qubit.qchannel.name and has_intersect_tmp_path_ids(epr.tmp_path_ids, path_ids)
        }

        # find another qubit to swap with
        candidates = (
            (q, v)
            for (q, v) in input
            if (q.qchannel is not None and q.qchannel.name in matched_channels)  # assigned to a matched channel
            and has_intersect_tmp_path_ids(epr.tmp_path_ids, v.tmp_path_ids)  # has overlapping tmp_path_ids
        )
        found = select_swap_qubit(self.fw._select_swap_qubit, qubit, epr, fib_entry, candidates)
        if not found:
            return None
        mq1, epr1 = found
        assert type(epr1) is WernerStateEntanglement

        chosen_path_id = random.choice(list(intersect_tmp_path_ids(epr, epr1)))
        if self.coordinated_decisions:
            epr.tmp_path_ids = epr1.tmp_path_ids = frozenset([chosen_path_id])
        fib_entry = self.fib.get(chosen_path_id)
        return mq1, fib_entry

    @override
    def swapping_succeeded(
        self,
        prev_epr: WernerStateEntanglement,
        next_epr: WernerStateEntanglement,
        new_epr: WernerStateEntanglement,
    ) -> None:
        new_epr.tmp_path_ids = intersect_tmp_path_ids(prev_epr, next_epr)

    @override
    def su_parallel_has_conflict(self, my_new_epr: WernerStateEntanglement, su_path_id: int) -> bool:
        assert my_new_epr.tmp_path_ids is not None
        if su_path_id not in my_new_epr.tmp_path_ids:
            assert not self.coordinated_decisions
            log.debug(f"{self.own}: Conflictual parallel swapping in statistical mux -> silently ignore")
            return True
        return False

    @override
    def su_parallel_succeeded(
        self, merged_epr: WernerStateEntanglement, new_epr: WernerStateEntanglement, other_epr: WernerStateEntanglement
    ) -> None:
        merged_epr.tmp_path_ids = intersect_tmp_path_ids(new_epr, other_epr)
