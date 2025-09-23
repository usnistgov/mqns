import random
from collections.abc import Callable

from typing_extensions import override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.node import QNode
from mqns.models.epr import WernerStateEntanglement
from mqns.network.proactive.fib import FibEntry
from mqns.network.proactive.mux_buffer_space import MuxSchemeFibBase
from mqns.network.proactive.mux_statistical import MuxSchemeDynamicBase, has_intersect_tmp_path_ids
from mqns.utils import log


def random_path_selector(fibs: list[FibEntry]) -> FibEntry:
    """
    Path selection strategy: random allocation.
    """
    return random.choice(fibs)


def select_weighted_by_swaps(fibs: list[FibEntry]) -> FibEntry:
    """
    Path selection strategy: swap-weighted allocation.
    """
    # Lower swaps = higher weight
    weights = [1.0 / (1 + len(e.swap)) for e in fibs]
    return random.choices(fibs, weights=weights, k=1)[0]


class MuxSchemeDynamicEpr(MuxSchemeDynamicBase, MuxSchemeFibBase):
    def __init__(
        self,
        name="dynamic EPR affection",
        *,
        path_select_fn: Callable[[list[FibEntry]], FibEntry] = random_path_selector,
    ):
        super().__init__(name)
        self.path_select_fn = path_select_fn

    @override
    def qubit_is_entangled(self, qubit: MemoryQubit, neighbor: QNode) -> None:
        possible_path_ids = self._qubit_is_entangled_0(qubit)
        if not possible_path_ids:  # all paths on the channel have been uninstalled
            return

        # TODO: if paths have different swap policies
        #       -> consider only paths for which this qubit may be eligible ??
        _, epr = self.memory.get(qubit.addr, must=True)
        assert isinstance(epr, WernerStateEntanglement)

        if epr.tmp_path_ids is None:
            # In principle, a random path_id is chosen for each elementary EPR during EPR generation.
            # The necessary information could be carried in the reservation message.
            # For ease of implementation, this choice is made at either primary or secondary node,
            # whichever receives the EPR notification earlier.
            fib_entries = [self.fib.get(pid) for pid in possible_path_ids]
            fib_entry = self.path_select_fn(fib_entries)
            epr.tmp_path_ids = frozenset([fib_entry.path_id])
        else:
            assert len(epr.tmp_path_ids) == 1
            fib_entry = self.fib.get(next(epr.tmp_path_ids.__iter__()))

        log.debug(f"{self.own}: qubit {qubit} has selected path_id {fib_entry.path_id}")

        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, neighbor)

    @override
    def select_eligible_qubit(self, mq0: MemoryQubit, fib_entry: FibEntry) -> MemoryQubit | None:
        assert mq0.path_id is None
        possible_path_ids = [fib_entry.path_id]
        mq1, _ = next(
            self.memory.find(
                lambda q, v: q.state == QubitState.ELIGIBLE  # in ELIGIBLE state
                and q.qchannel != mq0.qchannel  # assigned to a different channel
                and has_intersect_tmp_path_ids(v.tmp_path_ids, possible_path_ids),  # has compatible path_id
                has_epr=True,
            ),
            (None, None),
        )
        # TODO selection algorithm among found qubits
        return mq1

    @override
    def swapping_succeeded(
        self,
        prev_epr: WernerStateEntanglement,
        next_epr: WernerStateEntanglement,
        new_epr: WernerStateEntanglement,
    ) -> None:
        assert prev_epr.tmp_path_ids is not None
        assert next_epr.tmp_path_ids is not None
        assert prev_epr.tmp_path_ids == next_epr.tmp_path_ids
        new_epr.tmp_path_ids = prev_epr.tmp_path_ids

    @override
    def su_parallel_avoid_conflict(self, my_new_epr: WernerStateEntanglement, su_path_id: int) -> bool:
        assert my_new_epr.tmp_path_ids is not None
        if su_path_id not in my_new_epr.tmp_path_ids:
            raise Exception(f"{self.own}: Unexpected conflictual parallel swapping")
        return False

    @override
    def su_parallel_succeeded(
        self, merged_epr: WernerStateEntanglement, new_epr: WernerStateEntanglement, other_epr: WernerStateEntanglement
    ) -> None:
        assert new_epr.tmp_path_ids is not None
        assert other_epr.tmp_path_ids is not None
        assert new_epr.tmp_path_ids == other_epr.tmp_path_ids
        merged_epr.tmp_path_ids = new_epr.tmp_path_ids
