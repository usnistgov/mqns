"""
Proactive Forwarding
====================

This package implements proactive forwarding.

Multiplexing Schemes
--------------------

The proactive forwarder has a **multiplexing scheme** module that
determines how to allocate qubits to routing paths.
Every forwarder in the network must use the same multiplexing scheme,
which is also known by the controller.

Multiplexing schemes are implemented as :py:class:`MuxScheme` subclasses:

* :py:class:`MuxSchemeBufferSpace`
* :py:class:`MuxSchemeStatistical`
* :py:class:`MuxSchemeDynamicEpr`

Controller Architecture
-----------------------

:py:class:`ProactiveRoutingController` is the centralized controller application.

When the network uses buffer-space multiplexing scheme,
the :py:class:`~mqns.network.proactive.ctrl_ru.ResourceUtilization` data structure
keeps tracks of available qubits on each node.
The initial values are derived from topology construction.
They are then continuously updates as the controller makes routing decisions.
It is the controller's responsibility to ensure buffer-space qubit allocation does not
exceed available qubits on each node.

Forwarder Architecture
----------------------

:py:class:`ProactiveForwarder` is the data plane forwarder application.

The :py:class:`~mqns.network.proactive.fw_nb.ProactiveForwarderNorthbound` module
receives commands from the controller, updates the local FIB data structure,
and then informs the multiplexing scheme.

The multiplexing scheme communicates with :py:class:`~mqns.network.ll.LinkLayer`
to start/stop elementary entanglement generation.
For buffer-space, the qubit allocation comes from the controller as *multiplexing vector*.
Otherwise, the qubit allocation is statically configured as "channel capacity" during topology construction.
"""

from mqns.network.proactive.controller import NodeResourceInfo, ProactiveRoutingController
from mqns.network.proactive.forwarder import ProactiveForwarder, ProactiveForwarderInitKwargs
from mqns.network.proactive.mux import MuxScheme
from mqns.network.proactive.mux_buffer_space import MuxSchemeBufferSpace
from mqns.network.proactive.mux_dynamic_epr import MuxSchemeDynamicEpr
from mqns.network.proactive.mux_input import (
    MUX_SCHEME_MAP,
    MuxSchemeInput,
    MuxSchemeLiteral,
    mux_scheme_is_buffer_space,
    parse_mux_scheme,
)
from mqns.network.proactive.mux_statistical import MuxSchemeStatistical
from mqns.network.proactive.vora_swap import compute_vora_swap_sequence

__all__ = [
    "compute_vora_swap_sequence",
    "mux_scheme_is_buffer_space",
    "MUX_SCHEME_MAP",
    "MuxScheme",
    "MuxSchemeBufferSpace",
    "MuxSchemeDynamicEpr",
    "MuxSchemeInput",
    "MuxSchemeLiteral",
    "MuxSchemeStatistical",
    "NodeResourceInfo",
    "parse_mux_scheme",
    "ProactiveForwarder",
    "ProactiveForwarderInitKwargs",
    "ProactiveRoutingController",
]

for name in __all__:
    if name in (
        "MUX_SCHEME_MAP",
        "MuxSchemeInput",
        "MuxSchemeLiteral",
    ):
        continue
    globals()[name].__module__ = __name__
