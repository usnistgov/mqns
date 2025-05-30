from qns.models.delay.constdelay import ConstantDelayModel
from qns.models.delay.delay import DelayModel

DelayInput = float|DelayModel

def parseDelay(input: DelayInput) -> DelayModel:
    """
    Parse an argument that is either a DelayModel instance or a number that indicates constant delay in seconds.
    """
    return input if isinstance(input, DelayModel) else ConstantDelayModel(delay=input)
