from collections.abc import Callable, Iterator

from mqns.entity.memory import MemoryQubit
from mqns.entity.node import QNode
from mqns.models.epr import Entanglement
from mqns.network.fw import FibEntry
from mqns.utils import rng

type MemoryEprTuple = tuple[MemoryQubit, Entanglement]
type MemoryEprIterator = Iterator[MemoryEprTuple]

type SelectPurifQubit = (
    Callable[
        [MemoryQubit, FibEntry, QNode, list[MemoryEprTuple]],
        MemoryEprTuple,
    ]
    | None
)
"""
Qubit selection among purification candidates.
None means selecting the first candidate.
"""


def call_select_purif_qubit(
    fn: SelectPurifQubit,
    qubit: MemoryQubit,
    fib_entry: FibEntry,
    partner: QNode,
    candidates: MemoryEprIterator,
) -> MemoryEprTuple | None:
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
    candidates: list[MemoryEprTuple],
) -> MemoryEprTuple:
    _ = qubit, fib_entry, partner
    return candidates[rng.choice(len(candidates))]
