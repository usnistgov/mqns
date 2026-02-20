from mqns.network.fw.forwarder import Forwarder, ForwarderCounters

__all__ = [
    "Forwarder",
    "ForwarderCounters",
]

for name in __all__:
    globals()[name].__module__ = __name__
