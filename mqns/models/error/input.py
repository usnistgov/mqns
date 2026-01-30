from collections.abc import Callable
from typing import NotRequired, TypedDict

from mqns.models.error.error import ErrorModel, PerfectErrorModel

type ErrorModeConstructor = Callable[[], ErrorModel]


class ErrorModelInputPSurvival(TypedDict):
    p_survival: float


class ErrorModelInputPError(TypedDict):
    p_error: float


class ErrorModelInputTime(TypedDict):
    t: NotRequired[float]
    rate: float


class ErrorModelInputLength(TypedDict):
    length: NotRequired[float]
    rate: float


type ErrorModelInputDict = ErrorModelInputPSurvival | ErrorModelInputPError | ErrorModelInputTime | ErrorModelInputLength


def _apply_input_dict(error: ErrorModel, d: ErrorModelInputDict, dflt_t: float):
    p_survival = d.get("p_survival")
    if p_survival is not None:
        return error.set(p_survival=p_survival)

    p_error = d.get("p_error")
    if p_error is not None:
        return error.set(p_error=p_error)

    t, rate = d.get("t", d.get("length", dflt_t)), d.get("rate")
    if rate is not None:
        return error.set(t=t, rate=rate)

    return error


type ErrorModelInput = ErrorModel | tuple[ErrorModel | ErrorModeConstructor, ErrorModelInputDict] | ErrorModelInputDict | None
"""
Input to ``parse_error``.

Acceptable types:

* ``None``: ``PerfectErrorModel``.
* ``ErrorModel`` instance: used as is.
* Dict: used as probabilities with default error model type.
* ``ErrorModel`` instance and dict: clone and assign probabilities.
* ``ErrorModel`` type and dict: construct and assign probabilities.

Within the dict:

* ``{"p_survival":float}``: set survival probability.
* ``{"p_error":float}``: set error probability.
* ``{"t":float,"rate":float}``: set time based decay.
* ``{"length":float,"rate":float}``: set length based decay.
"""


def parse_error(input: ErrorModelInput, dflt: ErrorModeConstructor = PerfectErrorModel, dflt_t=0.0) -> ErrorModel:
    """
    Parse error model input.

    Args:
        input: input parameter.
        dflt: default ``ErrorModel`` subclass type.
        dflt_t: default ``t`` or ``length`` parameter.
    """
    if input is None:
        return PerfectErrorModel()
    if isinstance(input, ErrorModel):
        return input

    error, d = input if isinstance(input, tuple) else (dflt, input)
    error = error if isinstance(error, ErrorModel) else error()
    return _apply_input_dict(error, d, dflt_t)
