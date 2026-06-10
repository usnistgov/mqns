from collections.abc import Callable
from typing import cast, override

import numpy as np

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.node import QNode
from mqns.models.epr import Entanglement
from mqns.network.fw.fib import Fib, FibEntry
from mqns.network.fw.mux_buffer_space import MuxSchemeFibBase
from mqns.network.fw.mux_statistical import MuxSchemeDynamicBase
from mqns.network.fw.select import MemoryEprIterator
from mqns.utils import rng


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
    def qubit_is_entangled(self, mq: MemoryQubit, epr: Entanglement, neighbor: QNode) -> FibEntry | None:
        _ = neighbor
        # TODO: if paths have different swap policies
        #       -> consider only paths for which this qubit may be eligible ??

        if epr.affectionated_path_id < 0:
            # In principle, a random path_id is chosen for each elementary EPR during EPR generation.
            # The necessary information could be carried in the reservation message.
            # For ease of implementation, this choice is made at either primary or secondary node,
            # whichever receives the EPR notification earlier.
            selected_path = self._select_path(epr, self.fib, cast(list[int], mq.epr_path_ids))
            fib_entry = selected_path if type(selected_path) is FibEntry else self.fib.get(selected_path)
            epr.affectionated_path_id = fib_entry.path_id
        else:
            fib_entry = self.fib.get(epr.affectionated_path_id)

        mq.epr_path_ids = [fib_entry.path_id]
        mq.state = QubitState.PURIF
        return fib_entry

    @override
    def list_swap_candidates(self, mq0: MemoryQubit, fib_entry: FibEntry, input: MemoryEprIterator):
        _ = mq0
        return ((q, v) for (q, v) in input if fib_entry.path_id in cast(list[int], q.epr_path_ids))
