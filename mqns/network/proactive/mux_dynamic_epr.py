from collections.abc import Callable
from typing import Literal, override

import numpy as np

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.qchannel.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.fw import Fib, FibPath
from mqns.network.fw.select import call_select, parse_select, select_random
from mqns.network.proactive.mux_buffer_space import MuxSchemeFibBase
from mqns.network.proactive.mux_statistical import MuxSchemeDynamicBase
from mqns.utils import rng, unwrap, unwrap_cast


def _select_path_swap_weighted(path_ids: list[int], epr: Entanglement, fib: Fib) -> int:
    _ = epr
    # fewer swaps (shorter route) means higher weight
    weights = np.array([1.0 / (1 + len(fib.get_path(p).swap)) for p in path_ids])
    weights /= np.sum(weights)
    return path_ids[rng.choice(len(path_ids), p=weights)]


class MuxSchemeDynamicEpr(MuxSchemeFibBase, MuxSchemeDynamicBase):
    """
    Dynamic EPR Allocation multiplexing scheme.
    """

    type SelectPath = Callable[[list[int], Entanglement, Fib], int]
    """
    Path selection strategy.
    Function to select a path for an elementary entanglement.

    Args:
        epr: A newly established elementary EPR.
        fib: The FIB of the node making the selection.
        path_ids: List of candidate path IDs for this EPR.

    Returns:
        The selected path ID or FIB path entry.
    """

    SelectPath_random: SelectPath = select_random
    """
    Path selection strategy: random allocation.
    """

    SelectPath_swap_weighted: SelectPath = _select_path_swap_weighted
    """
    Path selection strategy: swap-weighted allocation.
    """

    def __init__(
        self,
        *,
        select_swap_qubit: MuxSchemeFibBase.SelectSwapQubitInput | None = None,
        select_path: SelectPath | Literal["random", "swap_weighted"] = SelectPath_random,
    ):
        """
        Args:
            select_swap_qubit: Function to select a qubit to swap with, default is first.
            select_path: Function to select a path for an entangled qubit, default is random.
        """
        super().__init__(select_swap_qubit)
        self._select_path = parse_select(type(self), "SelectPath_", select_path)

    @override
    def on_ll_attempt_success(self, ch: QuantumChannel, gid: int, epr: Entanglement) -> None:
        _ = gid
        paths = self.qchannel_paths_map.get(ch, [])
        chosen = call_select(paths, self._select_path, epr, self.fib)
        if chosen is not None:
            epr.affectionated_path_id = chosen

    @override
    def qubit_is_entangled(self, mq: MemoryQubit, epr: Entanglement) -> FibPath | None:
        if epr.affectionated_path_id == -1:
            return None
        mq.epr_path_ids = [epr.affectionated_path_id]
        mq.state = QubitState.PURIF
        return self.fib.get_path(epr.affectionated_path_id)

    @override
    def find_swap_with(self, mq0: MemoryQubit, epr0: Entanglement, fp: FibPath | None) -> tuple[MemoryQubit, FibPath] | None:
        fp = unwrap(fp)
        return self.select_swap_candidate(
            (mq0, epr0),
            fp,
            lambda q, _: (
                self.qubits_swappable(mq0, q)  # basic condition met
                and fp.path_id in unwrap_cast(q.epr_path_ids)  # has compatible path_id
            ),
        )
