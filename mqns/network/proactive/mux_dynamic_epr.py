from collections.abc import Callable
from typing import override

import numpy as np

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.node import QNode
from mqns.models.epr import Entanglement
from mqns.network.fw import Fib, FibEntry
from mqns.network.proactive.mux_buffer_space import MuxSchemeFibBase
from mqns.network.proactive.mux_statistical import MuxSchemeDynamicBase, has_intersect_tmp_path_ids
from mqns.network.proactive.select import MemoryEprIterator
from mqns.utils import log, rng


def _select_path_random(epr: Entanglement, fib: Fib, path_ids: list[int]) -> int:
    _ = epr, fib
    return rng.choice(path_ids)


def _select_path_swap_weighted(epr: Entanglement, fib: Fib, path_ids: list[int]) -> FibEntry:
    _ = epr
    entries = [fib.get(pid) for pid in path_ids]
    # fewer swaps (shorter route) means higher weight
    weights = np.array([1.0 / (1 + len(e.swap)) for e in entries])
    weights /= np.sum(weights)
    return entries[rng.choice(len(entries), p=weights)]


class MuxSchemeDynamicEpr(MuxSchemeFibBase, MuxSchemeDynamicBase):
    """
    Dynamic EPR Affection multiplexing scheme.
    """

    type SelectPath = Callable[[Entanglement, Fib, list[int]], int | FibEntry]
    """
    Path selection strategy.
    Function to select a path for an elementary entanglement.

    Args:
        epr: A newly established elementary EPR.
        fib: The FIB of the node making the selection.
        path_ids: List of candidate path IDs for this EPR.

    Returns:
        The selected path ID or FibEntry.
    """

    SelectPath_random: SelectPath = _select_path_random
    """
    Path selection strategy: random allocation.
    """

    SelectPath_swap_weighted: SelectPath = _select_path_swap_weighted
    """
    Path selection strategy: swap-weighted allocation.
    """

    def __init__(
        self,
        name="dynamic EPR affection",
        *,
        select_swap_qubit: MuxSchemeFibBase.SelectSwapQubit | None = None,
        select_path: SelectPath = SelectPath_random,
    ):
        """
        Args:
            select_swap_qubit: Function to select a qubit to swap with, default is first.
            select_path: Function to select a path for an entangled qubit, default is random.
        """
        super().__init__(name, select_swap_qubit)
        self._select_path = select_path

    @override
    def qubit_is_entangled(self, qubit: MemoryQubit, epr: Entanglement, neighbor: QNode) -> None:
        possible_path_ids = self._qubit_is_entangled_0(qubit)
        if not possible_path_ids:  # all paths on the channel have been uninstalled
            return

        # TODO: if paths have different swap policies
        #       -> consider only paths for which this qubit may be eligible ??

        if epr.tmp_path_ids is None:
            # In principle, a random path_id is chosen for each elementary EPR during EPR generation.
            # The necessary information could be carried in the reservation message.
            # For ease of implementation, this choice is made at either primary or secondary node,
            # whichever receives the EPR notification earlier.
            selected_path = self._select_path(epr, self.fib, possible_path_ids)
            fib_entry = selected_path if type(selected_path) is FibEntry else self.fib.get(selected_path)
            epr.tmp_path_ids = frozenset([fib_entry.path_id])
        else:
            assert len(epr.tmp_path_ids) == 1
            fib_entry = self.fib.get(next(epr.tmp_path_ids.__iter__()))

        log.debug(f"{self.node}: qubit {qubit} has selected path_id {fib_entry.path_id}")

        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, neighbor)

    @override
    def list_swap_candidates(self, mq0: MemoryQubit, fib_entry: FibEntry, input: MemoryEprIterator):
        assert mq0.path_id is None
        possible_path_ids = [fib_entry.path_id]
        return (
            (q, v)
            for (q, v) in input
            if has_intersect_tmp_path_ids(v.tmp_path_ids, possible_path_ids)  # has compatible path_id
        )

    @override
    def swapping_succeeded(self, prev_epr: Entanglement, next_epr: Entanglement, new_epr: Entanglement) -> None:
        assert prev_epr.tmp_path_ids is not None
        assert next_epr.tmp_path_ids is not None
        assert prev_epr.tmp_path_ids == next_epr.tmp_path_ids
        new_epr.tmp_path_ids = prev_epr.tmp_path_ids

    @override
    def su_parallel_has_conflict(self, my_new_epr: Entanglement, su_path_id: int) -> bool:
        assert my_new_epr.tmp_path_ids is not None
        if su_path_id not in my_new_epr.tmp_path_ids:
            raise Exception(f"{self.node}: Unexpected conflictual parallel swapping")
        return False

    @override
    def su_parallel_succeeded(self, merged_epr: Entanglement, new_epr: Entanglement, other_epr: Entanglement) -> None:
        assert new_epr.tmp_path_ids is not None
        assert other_epr.tmp_path_ids is not None
        assert new_epr.tmp_path_ids == other_epr.tmp_path_ids
        merged_epr.tmp_path_ids = new_epr.tmp_path_ids
