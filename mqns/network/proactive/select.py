import random
from collections.abc import Callable, Iterator

from mqns.entity.memory import MemoryQubit
from mqns.entity.node import QNode
from mqns.models.epr import WernerStateEntanglement
from mqns.network.proactive.fib import FibEntry

MemoryWernerTuple = tuple[MemoryQubit, WernerStateEntanglement]
MemoryWernerIterator = Iterator[MemoryWernerTuple]

SelectPurifQubit = (
    Callable[
        [MemoryQubit, FibEntry, QNode, list[MemoryWernerTuple]],
        MemoryWernerTuple,
    ]
    | None
)
"""
Qubit selection among purification candidates.
None means selecting the first candidate.
"""


def select_purif_qubit(
    fn: SelectPurifQubit,
    qubit: MemoryQubit,
    fib_entry: FibEntry,
    partner: QNode,
    candidates: MemoryWernerIterator,
) -> MemoryWernerTuple | None:
    if fn is None:
        return next(candidates, None)
    l = list(candidates)
    if len(l) == 0:
        return None
    return fn(qubit, fib_entry, partner, l)


def select_purif_qubit_random(
    qubit: MemoryQubit,
    fib_entry: FibEntry,
    partner: QNode,
    candidates: list[MemoryWernerTuple],
) -> MemoryWernerTuple:
    _ = qubit, fib_entry, partner
    return random.choice(candidates)


SelectSwapQubit = (
    Callable[
        [MemoryQubit, WernerStateEntanglement, FibEntry | None, list[MemoryWernerTuple]],
        MemoryWernerTuple,
    ]
    | None
)
"""
Qubit selection among swap candidates.
None means selecting the first candidate.
"""


def select_swap_qubit(
    fn: SelectSwapQubit,
    qubit: MemoryQubit,
    epr: WernerStateEntanglement,
    fib_entry: FibEntry | None,
    candidates: MemoryWernerIterator,
) -> MemoryWernerTuple | None:
    if fn is None:
        return next(candidates, None)
    l = list(candidates)
    if len(l) == 0:
        return None
    return fn(qubit, epr, fib_entry, l)


def select_swap_qubit_random(
    qubit: MemoryQubit,
    epr: WernerStateEntanglement,
    fib_entry: FibEntry | None,
    candidates: list[MemoryWernerTuple],
) -> MemoryWernerTuple:
    _ = qubit, epr, fib_entry
    return random.choice(candidates)


SelectPath = Callable[[list[FibEntry]], FibEntry]
"""
Path selection strategy, used in MuxSchemeDynamicEpr.
"""


def select_path_random(fibs: list[FibEntry]) -> FibEntry:
    """
    Path selection strategy: random allocation.
    """
    return random.choice(fibs)


def select_path_swap_weighted(fibs: list[FibEntry]) -> FibEntry:
    """
    Path selection strategy: swap-weighted allocation.
    """
    # Lower swaps = higher weight
    weights = [1.0 / (1 + len(e.swap)) for e in fibs]
    return random.choices(fibs, weights=weights, k=1)[0]
