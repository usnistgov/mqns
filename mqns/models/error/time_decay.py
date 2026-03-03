from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict

from mqns.models.error import ErrorModel
from mqns.models.error.input import ErrorModelConstructor, parse_error_str
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


def _time_decay_nop(target: "QuantumModel", t: Time) -> None:
    _ = target, t


time_decay_nop: TimeDecayFunc = _time_decay_nop
"""
TimeDecayFunc that does nothing.
"""


class TimeDecayDictRate(TypedDict):
    rate: float


class TimeDecayDictCohere(TypedDict):
    t_cohere: float


type TimeDecayInput = (
    TimeDecayDictRate | TimeDecayDictCohere | tuple[ErrorModelConstructor, TimeDecayDictRate | TimeDecayDictCohere] | str | None
)
"""``parse_time_decay`` input."""


def _set_rate(error: ErrorModel, value: float, accuracy: int):
    if value >= 0:  # value is decoherence rate in Hz
        rate = value / accuracy
    else:  # value is inverted coherence time in seconds
        rate = -1 / value / accuracy
    return error.set(t=0, rate=rate)


def parse_time_decay(input: TimeDecayInput, t_cohere: Time) -> TimeDecayFunc:
    """
    Parse TimeDecayFunc input.

    Args:
        input: input parameter.
        t_cohere: memory coherence time, used if ``input`` does not specify rate.

    Returns:
        TimeDecayFunc that accepts ``Time`` with same accuracy as ``t_cohere``.

    The input parameter could be one of:

    * ``None``: ``DephaseErrorModel`` with rate set to inverse of ``t_cohere``.
    * ``ErrorModel`` type: construct and assign rate as inverse of ``t_cohere``.
    * Dict: ``DephaseErrorModel`` with specified rate or t_cohere.
    * ``ErrorModel`` type and dict: construct and assign rate or t_cohere.
    * String: parsed following rules below.

    The dict could be one of:

    * ``{"rate":float}``: set given rate in Hz.
    * ``{"t_cohere":float}``: set rate as inverse of given t_cohere in seconds.

    The string could be one of:

    * ``"PERFECT"``: ``PerfectErrorModel``.
    * ``(DEPOLAR|DEPHASE|BITFLIP|DISSIPATION):rate`` (positive): construct with given rate in Hz.
    * ``(DEPOLAR|DEPHASE|BITFLIP|DISSIPATION):-t_cohere`` (negative): construct with
      rate set to inverse of given t_cohere in seconds.
    * The above models concatenated with ``:``, such as ``DEPOLAR:rate_depolar:DEPHASE:-t_dephase``.
    """
    if input is None:
        error = DephaseErrorModel().set(t=0, rate=1 / t_cohere.time_slot)
    elif isinstance(input, str):
        error = parse_error_str(input, "rate", (lambda m, v: _set_rate(m, v, t_cohere.accuracy)))
    else:
        ctor, d = input if isinstance(input, tuple) else (DephaseErrorModel, input)
        error = _set_rate(ctor(), d["rate"] if "rate" in d else -d["t_cohere"], t_cohere.accuracy)

    def apply_error_on(target: "QuantumModel", t: Time):
        error.set(t=t.time_slot)
        target.apply_error(error)

    return apply_error_on
