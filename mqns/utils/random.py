import os
import sys
from collections.abc import Sequence
from typing import Any, Literal, cast, overload

import numpy.random as npr

_rng = npr.default_rng()
"""
Real rng instance.
This may be re-assigned.
"""

_FAST_METHODS = (
    "choice",
    "exponential",
    "geometric",
    "integers",
    "normal",
    "random",
    "shuffle",
    "uniform",
)
"""
Method names on ``rng`` to avoid getattr overhead.
"""

_ENV_NAME = "MQNS_SEED"


@overload
def seed_env() -> int | None:
    """
    Parse random seed from MQNS_SEED environment variable.

    Returns:
        Integer random seed, or ``None`` if unset.
    """


@overload
def seed_env(dflt: int) -> int:
    """
    Parse random seed from MQNS_SEED environment variable.

    Returns:
        Integer random seed, or ``dflt`` if unset.
    """


def seed_env(dflt: int | None = None):
    env = os.getenv(_ENV_NAME)
    if env is None:
        return dflt
    try:
        return int(env)
    except ValueError:
        raise ValueError(f"{_ENV_NAME} environment variable is set but not an integer")


def seed_seq_env(n_runs: int, dflt_base: int) -> Sequence[int]:
    """
    Generate a sequence of random seeds for consecutive experiment runs.
    The first seed is parsed from MQNS_SEED environment variable if present.

    Args:
        n_runs: How many experiment runs.
        dflt_base: The first seed, if MQNS_SEED environment variable is unset.

    Returns:
        An increasing sequence of integers.
    """
    seed_base = seed_env(dflt_base)
    return range(seed_base, seed_base + n_runs)


_next_seed_env: int | None = None


def _reseed_env() -> int | None:
    global _next_seed_env

    if _next_seed_env is not None:
        seed = _next_seed_env
    else:
        seed = seed_env()
        if seed is None:
            return None

    if "pytest" in sys.modules:
        print(f"{_ENV_NAME}={seed}")

    _next_seed_env = seed + 1
    return seed


class RngUtils:
    def reseed(self, seed: int | None | Literal["env"]):
        """
        Reseed the random number generator.

        The new ``seed`` could be one of:

        * Integer seed.
        * ``None`` for NumPy default.
        * "env": set the first seed with MQNS_SEED environment variable,
          auto-increment subsequent seeds.
          If MQNS_SEED is unset, this is equivalent to ``None``.

        When "env" mode is used with pytest, the actual seed value is printed to stdout.
        Combined with ``pytest-repeat``, this enables detecting seed-specific test failures.
        To run a test case repeatedly with increasing seeds:

            MQNS_SEED=0 pytest test_file.py::test_case --count=100 -x

        To run a test case with a specific seed:

            MQNS_SEED=47 pytest test_file.py::test_case
        """
        global _rng
        if seed == "env":
            seed = _reseed_env()

        _rng = npr.default_rng(npr.PCG64(seed))

        if isinstance(self, RngProxy):
            self._update_fast_methods()


class RngProxy(RngUtils):
    """
    Proxy class for global rng.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_rng, name)

    def _update_fast_methods(self) -> None:
        for method in _FAST_METHODS:
            self.__dict__[method] = getattr(_rng, method)


class RngPublic(npr.Generator, RngUtils):
    """
    Global random number generator, public API declaration.
    """


rng = cast(RngPublic, RngProxy())
"""
Global random number generator.
"""

cast(RngProxy, rng)._update_fast_methods()
