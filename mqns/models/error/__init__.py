from mqns.models.error.chain import ChainErrorModel
from mqns.models.error.coherent import CoherentErrorModel
from mqns.models.error.dissipation import DissipationErrorModel
from mqns.models.error.error import ErrorModel, PerfectErrorModel
from mqns.models.error.pauli import BitFlipErrorModel, DephaseErrorModel, DepolarErrorModel, PauliErrorModel
from mqns.models.error.time_decay import TimeDecayFunc, TimeDecayInput, parse_time_decay, time_decay_nop

__all__ = [
    "BitFlipErrorModel",
    "ChainErrorModel",
    "CoherentErrorModel",
    "DephaseErrorModel",
    "DepolarErrorModel",
    "DissipationErrorModel",
    "ErrorModel",
    "parse_time_decay",
    "PauliErrorModel",
    "PerfectErrorModel",
    "time_decay_nop",
    "TimeDecayFunc",
    "TimeDecayInput",
]

for name in __all__:
    if name in ("TimeDecayFunc", "TimeDecayInput"):
        continue
    globals()[name].__module__ = __name__
