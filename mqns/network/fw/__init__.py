from mqns.network.fw.classic import fw_control_cmd_handler, fw_signaling_cmd_handler
from mqns.network.fw.fib import Fib, FibEntry
from mqns.network.fw.forwarder import Forwarder, ForwarderCounters
from mqns.network.fw.message import MultiplexingVector, SwapSequence

__all__ = [
    "Fib",
    "FibEntry",
    "Forwarder",
    "ForwarderCounters",
    "fw_control_cmd_handler",
    "fw_signaling_cmd_handler",
    "MultiplexingVector",
    "SwapSequence",
]

for name in __all__:
    if name in ("MultiplexingVector", "SwapSequence"):
        continue
    globals()[name].__module__ = __name__
