from collections.abc import Callable
from typing import TYPE_CHECKING

from mqns.models.error.input import ErrorModelConstructor, ErrorModelInput, parse_error
from mqns.models.error.pauli import DephaseErrorModel
from mqns.simulator import Time

if TYPE_CHECKING:
    from mqns.models.core import QuantumModel

type TimeDecayFunc = Callable[["QuantumModel", Time], None]
"""
Function to apply time based decay.

Args:
    [0]: target qubit or EPR.
    [1]: duration since last fidelity update.
"""


def time_decay_nop(target: "QuantumModel", t: Time) -> None:
    """
    TimeDecayFunc that does nothing.
    """
    _ = target, t


def make_time_decay_func(
    input: ErrorModelInput = None,
    *,
    t_cohere: Time,
    dflt: ErrorModelConstructor = DephaseErrorModel,
) -> TimeDecayFunc:
    """
    Build TimeDecayFunc from coherence time.

    Args:
        input: ``ErrorModel`` instance or type.
               If ``None``, ``dflt`` is used.
        t_cohere: coherence time, which is the inverse of coherence rate.
        dflt: default error mode type, defaults to dephasing.

    Returns:
        TimeDecayFunc that accepts ``Time`` with same accuracy as ``t_cohere``.
    """
    error = dflt() if input is None else parse_error(input, dflt)
    error.set(t=0, rate=1 / t_cohere.time_slot)

    def apply_error_on(target: "QuantumModel", t: Time):
        error.set(t=t.time_slot)
        target.apply_error(error)

    return apply_error_on
