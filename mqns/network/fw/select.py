from collections.abc import Callable, Iterable, Sequence
from typing import Any, Never, cast

from mqns.entity.memory import MemoryQubit
from mqns.models.epr import Entanglement
from mqns.utils import rng


def parse_select[F: Callable, N: None | Never](cls: type, prefix: str, input: F | str | N) -> F | N:
    """
    Parse candidate selection function from string keyword.

    Args:
        cls: Class where predefined candidate selection functions are defined.
        prefix: Prefix of predefined candidate selection functions on ``cls``.
        input: Input parameter; its string possibilities should be constrained with :py:class:`typing.Literal`.

    Returns:
        Predefined or custom candidate selection function.
    """
    if callable(input) or input is None:
        return input
    f = getattr(cls, f"{prefix}{input}")
    assert callable(f)
    return cast(F, f)


def call_select[T, R](candidates: Iterable[T], fn: Callable[..., R] | None, *args: Any) -> T | R | None:
    """
    Call candidate selection function.

    Args:
        candidates: Iterator of candidates.
        fn: Selection function or None, ``fn(candidates: list[T], *args)``.

    Returns:
        Chosen candidate or ``None`` for empty input.
    """
    if not fn:
        return next(iter(candidates), None)
    l: Sequence[T] = candidates if isinstance(candidates, Sequence) else list(candidates)
    if not l:
        return None
    if len(l) == 1:
        return l[0]
    return fn(l, *args)


def select_random[T](candidates: Sequence[T], *_: Any) -> T:
    """
    Candidate selection function that selects a random candidate with uniform probability.
    """
    return candidates[rng.choice(len(candidates))]


type MemoryEprTuple = tuple[MemoryQubit, Entanglement]


def select_swap_qubit_oldest(candidates: Sequence[MemoryEprTuple], *_) -> MemoryEprTuple:
    return min(candidates, key=lambda c: c[0].eligible_time)


def select_swap_qubit_newest(candidates: Sequence[MemoryEprTuple], *_) -> MemoryEprTuple:
    return max(candidates, key=lambda c: c[0].eligible_time)
