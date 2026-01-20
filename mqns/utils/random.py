from collections.abc import Callable
from typing import Any, override

import numpy.random as npr

rng = npr.default_rng()


def set_seed(seed: int | None):
    """
    Reseed the random number generator.
    """
    global rng
    rng = npr.default_rng(npr.PCG64(seed))


class FixedRng(npr.Generator):
    """
    Random number generator that returns fixed values.

    This is primarily useful for unit testing.
    """

    def __init__(self, v: Callable[[], float] | float | None = None):
        super().__init__(rng.bit_generator)
        self._v = (lambda: v) if isinstance(v, (int, float)) else v

    @override
    def random(self, *args, **kwargs) -> Any:
        return self._v() if self._v else super().random(*args, **kwargs)

    @override
    def uniform(self, *args, **kwargs) -> Any:
        return self._v() if self._v else super().uniform(*args, **kwargs)
