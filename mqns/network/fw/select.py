from collections.abc import Callable, Iterable, Iterator
from typing import Any

from mqns.entity.memory import MemoryQubit
from mqns.entity.node import QNode
from mqns.models.epr import Entanglement
from mqns.network.fw import FibEntry
from mqns.utils import rng


def call_select[T, R](candidates: Iterable[T], fn: Callable[..., R] | None, *args: Any) -> T | R | None:
    """
    Call candidate selection function.

    Args:
        candidates: Iterator of candidates.
        fn: Selection function or None, ``fn(*args, candidates: list[T])``.

    Returns:
        Chosen candidate or ``None`` for empty input.
    """
    if not fn:
        return next(iter(candidates), None)
    l: list[T] = candidates if isinstance(candidates, list) else list(candidates)
    if not l:
        return None
    if len(l) == 1:
        return l[0]
    return fn(l, *args)


def select_random[T](candidates: list[T], *_: Any) -> T:
    """
    Candidate selection function that selects a random candidate with uniform probability.
    """
    return candidates[rng.choice(len(candidates))]


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
