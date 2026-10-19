"""
This package provides :py:class:`NetworkBuilder`, an orchestrator of quantum network simulation.
"""

from mqns.network.builder.builder import (
    CTRL_DELAY,
    AppsForwarderArgs,
    ChannelArgs,
    ChannelDef,
    ChannelParam,
    NetworkBuilder,
    NodeArgs,
    NodeDef,
    TopoCommonArgs,
)
from mqns.network.builder.tap import tap_configure
from mqns.network.builder.topogen import GridTopoGen, LinearTopoGen, TopoGen, TreeTopoGen

__all__ = [
    "AppsForwarderArgs",
    "ChannelArgs",
    "ChannelDef",
    "ChannelParam",
    "CTRL_DELAY",
    "GridTopoGen",
    "LinearTopoGen",
    "NetworkBuilder",
    "NodeArgs",
    "NodeDef",
    "tap_configure",
    "TopoCommonArgs",
    "TopoGen",
    "TreeTopoGen",
]

for name in __all__:
    if name in ("CTRL_DELAY",):
        continue
    globals()[name].__module__ = __name__
