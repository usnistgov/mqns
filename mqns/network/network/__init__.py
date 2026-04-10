from mqns.network.network.network import QuantumNetwork
from mqns.network.network.request import Request, RequestAttr
from mqns.network.network.timing import TimingMode, TimingModeAsync, TimingModeSync, TimingPhase, TimingPhaseEvent

__all__ = [
    "QuantumNetwork",
    "Request",
    "RequestAttr",
    "TimingMode",
    "TimingModeAsync",
    "TimingModeSync",
    "TimingPhase",
    "TimingPhaseEvent",
]

for name in __all__:
    globals()[name].__module__ = __name__
