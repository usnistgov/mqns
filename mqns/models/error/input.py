import copy
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Never, NotRequired, TypedDict, cast

from mqns.models.error.chain import ChainErrorModel
from mqns.models.error.dissipation import DissipationErrorModel
from mqns.models.error.error import ErrorModel, PerfectErrorModel
from mqns.models.error.pauli import BitFlipErrorModel, DephaseErrorModel, DepolarErrorModel

type ErrorModelConstructor = Callable[[], ErrorModel]


class ErrorModelDictPSurvival(TypedDict):
    p_survival: float


class ErrorModelDictPError(TypedDict):
    p_error: float


class ErrorModelDictTime(TypedDict):
    t: NotRequired[float]
    rate: float


class ErrorModelDictLength(TypedDict):
    length: NotRequired[float]
    rate: float


def _apply_input_dict(
    error: ErrorModel,
    d: dict[Never, Never] | ErrorModelDictPSurvival | ErrorModelDictPError | ErrorModelDictTime | ErrorModelDictLength,
    dflt_t: float,
):
    if (p_survival := cast(ErrorModelDictPSurvival, d).get("p_survival")) is not None:
        return error.set(p_survival=p_survival)

    if (p_error := cast(ErrorModelDictPError, d).get("p_error")) is not None:
        return error.set(p_error=p_error)

    if (rate := cast(ErrorModelDictTime | ErrorModelDictLength, d).get("rate")) is not None:
        t = cast(ErrorModelDictTime, d).get("t", cast(ErrorModelDictLength, d).get("length", dflt_t))
        return error.set(t=t, rate=rate)

    return error


_STR_ERROR_TYPES: dict[str, ErrorModelConstructor] = {
    "BITFLIP": BitFlipErrorModel,
    "DEPHASE": DephaseErrorModel,
    "DEPOLAR": DepolarErrorModel,
    "DISSIPATION": DissipationErrorModel,
}

_STR_PARSE_ERROR = f"unrecognized ErrorModelInput string: PERFECT | {'{'}{'|'.join(_STR_ERROR_TYPES)}{'}'}"


type ParseErrorStrSetter = Callable[[ErrorModel, float], Any]


def _parse_error_str(input: Sequence[str], value_desc: str, set_onto: ParseErrorStrSetter) -> Iterable[ErrorModel]:
    it = iter(input)
    for token in it:
        if token == "PERFECT":
            yield PerfectErrorModel()
            continue

        try:
            m = _STR_ERROR_TYPES[token]()
            value = float(next(it))
        except (KeyError, StopIteration, ValueError):
            raise ValueError(f"{_STR_PARSE_ERROR}:{value_desc}(float)")

        set_onto(m, value)
        yield m


def parse_error_str(input: str, value_desc: str, set_onto: ParseErrorStrSetter) -> ErrorModel:
    """
    Parse error model from string input.

    Args:
        input: input string, a sequence of tokens delimited by ``:``, where each token
               either identifies an error model type or is a float value.
        value_desc: description of each float value.
        set_onto: callback function to save the float value onto constructed ``ErrorModel``.

    Returns:
        ErrorModel, either singular subclass or ``ChainErrorModel``.
    """
    chain = list(_parse_error_str(input.split(":"), value_desc, set_onto))
    if len(chain) == 1:
        return chain[0]
    return ChainErrorModel(chain)


type ErrorModelInput[D: ErrorModelDictTime | ErrorModelDictLength] = (
    ErrorModel
    | tuple[ErrorModel, dict[Never, Never] | ErrorModelDictPSurvival | ErrorModelDictPError | D]
    | tuple[ErrorModel | ErrorModelConstructor, ErrorModelDictPSurvival | ErrorModelDictPError | D]
    | ErrorModelDictPSurvival
    | ErrorModelDictPError
    | D
    | str
    | None
)

type ErrorModelInputBasic = ErrorModelInput[Never]
"""``parse_error`` input, accepting dict with probability."""
type ErrorModelInputTime = ErrorModelInput[ErrorModelDictTime]
"""``parse_error`` input, accepting dict with time-based decay."""
type ErrorModelInputLength = ErrorModelInput[ErrorModelDictLength]
"""``parse_error`` input, accepting dict with length-based decay."""


def parse_error(
    input: ErrorModelInputBasic | ErrorModelInputTime | ErrorModelInputLength, dflt: ErrorModelConstructor, dflt_t: float
) -> ErrorModel:
    """
    Parse error model input.

    Args:
        input: input parameter.
        dflt: default ``ErrorModel`` subclass type.
        dflt_t: default ``t`` or ``length`` parameter; -1 if time/length based decay is unsupported.

    The input parameter could be one of:

    * ``None``: ``PerfectErrorModel``.
    * ``ErrorModel`` instance: clone.
    * Dict: used as probabilities with default error model type.
    * ``ErrorModel`` instance and dict: clone and assign probabilities.
    * ``ErrorModel`` type and dict: construct and assign probabilities.
    * String: parsed following rules below.

    The dict could be one of:

    * ``{"p_survival":float}``: set survival probability.
    * ``{"p_error":float}``: set error probability.
    * ``{"t":float,"rate":float}``: set time based decay; ``dflt_t`` is used if ``t`` is omitted.
    * ``{"length":float,"rate":float}``: set length based decay; ``dflt_t`` is used if ``length`` is omitted.

    The string could be one of:

    * ``"PERFECT"``: ``PerfectErrorModel``.
    * ``(DEPOLAR|DEPHASE|BITFLIP|DISSIPATION):rate``: construct with specified rate,
      available for memory / qchannel errors with time/length based decay (``dflt_t>=0``).
    * ``(DEPOLAR|DEPHASE|BITFLIP|DISSIPATION):p_error``: construct with
      specified error probability, available for BSA / operate / measure errors (``dflt_t<0``).
    * The above models concatenated with ``:``, such as ``DEPOLAR:rate_depolar:DEPHASE:rate_dephase``.
    """
    if input is None:
        return PerfectErrorModel()
    if isinstance(input, ErrorModel):
        return copy.deepcopy(input)
    if isinstance(input, str):
        return parse_error_str(
            input,
            "p_error" if dflt_t < 0 else "rate",
            (lambda m, v: m.set(p_error=v)) if dflt_t < 0 else (lambda m, v: m.set(t=dflt_t, rate=v)),
        )

    base, d = input if isinstance(input, tuple) else (dflt, input)
    error = copy.deepcopy(base) if isinstance(base, ErrorModel) else base()
    return _apply_input_dict(error, d, dflt_t)
