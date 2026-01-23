from mqns.models.delay.constdelay import ConstantDelayModel
from mqns.models.delay.delay import DelayModel

DelayInput = float | DelayModel
"""
Input to ``parse_delay``.

Acceptable types:
* ``DelayModel`` instance: used as is.
* Number: constant delay in seconds.
"""


def parse_delay(input: DelayInput) -> DelayModel:
    """Parse delay model input."""
    return input if isinstance(input, DelayModel) else ConstantDelayModel(delay=input)
