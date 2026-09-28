from collections.abc import Mapping
from typing import Literal

from mqns.models.epr.entanglement import Entanglement
from mqns.models.epr.mixed import MixedStateEntanglement
from mqns.models.epr.qubit_pair import EntangledQubitPair
from mqns.models.epr.werner import WernerStateEntanglement

type EprTypeLiteral = Literal["W", "M", "Q"]
"""
String representation of commonly used entanglement models.
"""

EPR_TYPE_MAP: Mapping[EprTypeLiteral, type[Entanglement]] = {
    "W": WernerStateEntanglement,
    "M": MixedStateEntanglement,
    "Q": EntangledQubitPair,
}

type EprTypeInput = type[Entanglement] | EprTypeLiteral | None
"""
Entanglement model input parsable by ``parse_epr_type``.

* ``None`` or ``"W"``: Werner state.
* ``"M"``: Bell-diagonal state.
* ``"Q"``: Entangled qubit pair.
* ``Entanglement`` subclass constructor: use the given type.
"""


def parse_epr_type(input: EprTypeInput) -> type[Entanglement]:
    """
    Parse an entanglement model input.
    """
    if input is None:
        return WernerStateEntanglement

    if isinstance(input, str):
        return EPR_TYPE_MAP[input]

    return input
