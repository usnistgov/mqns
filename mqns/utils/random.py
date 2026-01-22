from typing import Any, cast

import numpy.random as npr

_rng = npr.default_rng()
"""
Real rng instance.
This may be re-assigned.
"""


class RngUtils:
    def reseed(self, seed: int | None):
        """
        Reseed the random number generator.
        """
        global _rng
        _rng = npr.default_rng(npr.PCG64(seed))


class RngProxy(RngUtils):
    """
    Proxy class for global rng.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_rng, name)


class RngPublic(npr.Generator, RngUtils):
    """
    Global random number generator, public API declaration.
    """


rng = cast(RngPublic, RngProxy())
"""
Global random number generator.
"""
