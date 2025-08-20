import random
from collections.abc import Callable

from qns.entity.memory import MemoryQubit, QubitState
from qns.entity.node import QNode
from qns.models.epr import WernerStateEntanglement
from qns.network.proactive.fib import FIBEntry
from qns.network.proactive.mux_buffer_space import MuxSchemeFibBase
from qns.network.proactive.mux_statistical import MuxSchemeDynamicBase

try:
    from typing import override
except ImportError:
    from typing_extensions import override


def random_path_selector(fibs: list[FIBEntry]) -> int:
    """
    Path selection strategy: random allocation.
    """
    return random.choice(fibs)["path_id"]


def select_weighted_by_swaps(fibs: list[FIBEntry]) -> int:
    """
    Path selection strategy: swap-weighted allocation.
    """
    # Lower swaps = higher weight
    weights = [1.0 / (1 + len(e["swap_sequence"])) for e in fibs]
    return random.choices(fibs, weights=weights, k=1)[0]["path_id"]


class MuxSchemeDynamicEpr(MuxSchemeDynamicBase, MuxSchemeFibBase):
    def __init__(
        self,
        name="dynamic EPR affection",
        *,
        path_select_fn: Callable[[list[FIBEntry]], int] = random_path_selector,
    ):
        super().__init__(name)
        self.path_select_fn = path_select_fn

    @override
    def qubit_is_entangled(self, qubit: MemoryQubit, neighbor: QNode) -> None:
        possible_path_ids = self._qubit_is_entangled_0(qubit)
        # TODO: if paths have different swap policies
        #       -> consider only paths for which this qubit may be eligible ??
        _, epr = self.memory.get(address=qubit.addr, must=True)
        assert isinstance(epr, WernerStateEntanglement)
        if epr.tmp_path_ids is None:  # whatever neighbor is first
            fib_entries = [self.fib.get_entry(pid, must=True) for pid in possible_path_ids]
            path_id = self.path_select_fn(fib_entries)
            epr.tmp_path_ids = frozenset([path_id])

        fib_entry = self.fib.get_entry(next(epr.tmp_path_ids.__iter__()), must=True)
        self.own.get_qchannel(neighbor)  # ensure qchannel exists
        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, neighbor)

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
