import functools
import itertools
from collections.abc import Sequence
from typing import Literal, NotRequired, Self, TypedDict, Unpack, cast, overload

from tap import Tap

from mqns.entity.memory import QuantumMemoryInitKwargs
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import (
    LinkArch,
    LinkArchDimBk,
    LinkArchDimBkSeq,
    LinkArchDimDual,
    LinkArchSim,
    LinkArchSr,
    QuantumChannel,
)
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.models.error import TimeDecayInput
from mqns.models.error.input import ErrorModelInputBasic, ErrorModelInputLength, ErrorModelInputTime
from mqns.network.fw import (
    ForwarderInitKwargs,
    MuxSchemeBufferSpace,
    QubitAllocationType,
    RoutingPath,
    RoutingPathInitArgs,
    RoutingPathMulti,
    RoutingPathSingle,
    SwapPolicy,
)
from mqns.network.network import QuantumNetwork, TimingMode, TimingModeAsync, TimingModeSync
from mqns.network.proactive import ProactiveForwarder, ProactiveRoutingController
from mqns.network.protocol.classicbridge import ClassicBridge
from mqns.network.protocol.link_layer import LinkLayer
from mqns.network.reactive import ReactiveForwarder, ReactiveRoutingController
from mqns.network.route import DijkstraRouteAlgorithm, RouteAlgorithm, YenRouteAlgorithm
from mqns.network.topology import ClassicTopology, Topology
from mqns.network.topology.customtopo import CustomTopology, Topo, TopoController, TopoQChannel, TopoQNode

type NodePair = str | tuple[str, str]
"""
Two node names on a channel or routing path.
This could be either a tuple of two node names, or a string delimited by hyphen (``-``).
"""

type EprTypeLiteral = Literal["W", "M"]
"""
String representation of commonly used entanglement models.
"""
EPR_TYPE_MAP: dict[EprTypeLiteral, type[Entanglement]] = {
    "W": WernerStateEntanglement,
    "M": MixedStateEntanglement,
}

type LinkArchLiteral = Literal["DIM-BK", "DIM-BK-SeQUeNCe", "DIM-dual", "SR", "SIM"]
"""
String representation of commonly used link architectures.
"""
LINK_ARCH_MAP: dict[LinkArchLiteral, type[LinkArch]] = {
    "DIM-BK": LinkArchDimBk,
    "DIM-BK-SeQUeNCe": LinkArchDimBkSeq,
    "DIM-dual": LinkArchDimDual,
    "SR": LinkArchSr,
    "SIM": LinkArchSim,
}
type LinkArchDef = LinkArch | type[LinkArch] | LinkArchLiteral


def tap_configure(tap: Tap) -> None:
    """
    When called from ``Tap.configure()`` function, define command line arguments for supported literal types.

    Recognized keys:

    * ``mode``
    * ``sync_timing``

    Recognized types:

    * ``EprTypeLiteral``
    * ``LinkArchLiteral``
    * ``ErrorModelInput*``
    * ``TimeDecayInput``
    """
    for key, typ in tap._annotations.items():
        if key == "mode":
            tap.add_argument(
                f"--{key}",
                help=f"(default={getattr(tap, key)}) choose mode: [P]roactive/[R]eactive forwarding, "
                "[C]entralized/[D]istributed control, [A]sync/[S]ync timing",
            )
        elif key == "sync_timing":
            dflt = getattr(tap, key, [])
            dflt_desc = f"default={dflt}" if dflt else "default: derive from t_cohere"
            tap.add_argument(
                f"--{key}",
                type=float,
                nargs=3,
                default=dflt,
                metavar=("t_ext", "t_rtg", "t_int"),
                help=f"(3*float, {dflt_desc}) SYNC timing mode phase durations in seconds",
            )
        elif typ is EprTypeLiteral:
            tap.add_argument(f"--{key}", type=str, default="W", choices=EPR_TYPE_MAP.keys())
        elif typ is LinkArchLiteral:
            tap.add_argument(f"--{key}", type=str, default="DIM-BK-SeQUeNCe", choices=LINK_ARCH_MAP.keys())
        elif typ is ErrorModelInputBasic:
            tap.add_argument(f"--{key}", type=str, metavar="ErrorModelType:p_error")
        elif typ in (ErrorModelInputTime, ErrorModelInputLength, TimeDecayInput):
            tap.add_argument(f"--{key}", type=str, metavar="ErrorModelType:rate")


CTRL_DELAY = 5e-06
"""
Delay of the classic channels between the controller and each QNode, in seconds.

In most examples, the overall simulation duration is increased by this value,
so that the QNodes can perform entanglement forwarding for the full intended duration.
"""


class NodeArgs(TypedDict, total=False):
    mem_capacity: int
    """Memory capacity, defaults to ``-1`` for deriving from channel capacities."""
    t_cohere: float
    """Memory coherence time in seconds, defaults to ``0.02``."""
    memory_decay: TimeDecayInput
    """Memory time decay function, defaults to dephasing in ``t_cohere``."""


class NodeDef:
    """Node definition."""

    def __init__(self, name: str, **kwargs: Unpack[NodeArgs]):
        self.name = name
        self.d = kwargs


class ChannelArgs(TypedDict, total=False):
    ch_length: float
    """
    Channel length in kilometer, defaults to ``1.0``.
    """
    ch_capacity: int | tuple[int, int]
    """
    Channel capacity, defaults to ``1``.
    An integer applies to both sides; a tuple applies to ``(left,right)`` sides.
    """
    link_arch: LinkArchDef
    """
    Link architecture, defaults to ``LinkArchDimBkSeq``.
    """
    fiber_alpha: float
    """
    Fiber loss in dB/km, defaults to ``0.2``.
    This determines success probability.
    """
    fiber_error: ErrorModelInputLength
    """
    Fiber error model, defaults to depolarizing with ``0.01`` error probability.
    This determines qualify of entangled state.
    """
    bsa_error: ErrorModelInputBasic
    """
    Photonic Bell-state analyzer or absorptive memory capture error model, defaults to perfect.
    This determines qualify of entangled state.
    """


class ChannelParam:
    """Channel parameters."""

    def __init__(self, **kwargs: Unpack[ChannelArgs]):
        self.d = kwargs


class ChannelDef(ChannelParam):
    """Channel definition."""

    def __init__(self, np: NodePair, /, **kwargs: Unpack[ChannelArgs]):
        super().__init__(**kwargs)
        self.np = np


class TopoCommonArgs(NodeArgs, ChannelArgs):
    """
    Combination of ``NodeArgs`` and ``ChannelArgs``, with additional parameters.
    """

    # Conceptually these should belong to either NodeArgs or ChannelArgs,
    # but implementation limitation made them non-configurable.
    # If use case arises, these could be refactored to be per-node or per-channel.
    init_fidelity: NotRequired[float | Sequence[float] | None]
    """
    Fidelity of generated entangled pairs, defaults to ``0.99``.
    If ``None``, determine with error models in link architecture.
    """
    entg_attempt_rate: NotRequired[float]
    """Maximum entanglement attempts per second, defaults to ``50_000_000`` but currently ineffective."""
    eta_d: NotRequired[float]
    """Detector efficiency, defaults to ``0.95``."""
    eta_s: NotRequired[float]
    """Source efficiency, defaults to ``0.95``."""
    frequency: NotRequired[float]
    """Entanglement source frequency, defaults to ``1_000_000``."""


class AppsCommonArgs(TypedDict, total=False):
    timing: TimingMode | Sequence[float] | None
    """
    Network timing mode, defaults to ASYNC.
    If specified as three floats, construct ``TimingModeSync`` with these durations.
    """


class AppsForwarderArgs(AppsCommonArgs, ForwarderInitKwargs):
    """
    Combination of AppsCommonArgs and ForwarderInitKwargs.

    Args:
        p_swap: Probability of successful entanglement swapping in forwarder, defaults to ``0.5``.
    """


def _split_node_pair(np: NodePair) -> tuple[str, str]:
    if isinstance(np, str):
        tokens = np.split("-")
        if len(tokens) != 2:
            raise ValueError(f"expect two node names in '{np}'")
        return cast(tuple[str, str], tuple(tokens))
    return np


class NetworkBuilder:
    """
    Orchestrator of quantum network simulation with LinkLayer and forwarder.

    Usage:

    1. Call one ``.topo*()`` method to define topology shape.
    2. Call one ``.{proactive|reactive}_{centralized|distributed}()`` method to choose applications.
    3. Call ``.request()`` method to define end-to-end requests or routing paths.
    4. Call ``.make_network()`` method to construct ``QuantumNetwork`` ready for simulation.
    """

    def __init__(
        self,
        *,
        route: RouteAlgorithm[QNode, QuantumChannel] = DijkstraRouteAlgorithm(),
        epr_type: type[Entanglement] | EprTypeLiteral = "W",
    ):
        """
        Constructor.

        Args:
            route: Route algorithm, defaults to Dijkstra.
            epr_type: Network-wide EPR model, defaults to Werner state.
        """

        self.route = route
        self.epr_type = epr_type if isinstance(epr_type, type) else EPR_TYPE_MAP[epr_type]

        self.qnodes: list[TopoQNode] = []
        self.qnode_by_name: dict[str, TopoQNode] = {}
        self.extensible_memory_by_name: dict[str, QuantumMemoryInitKwargs] = {}
        self.qnode_apps: list[Application] = []
        self.qchannels: list[TopoQChannel] = []
        self.controller_apps: list[Application] = []

        self.qubit_allocation = QubitAllocationType.DISABLED
        self.requests: list[tuple[str, str]] = []

    def _save_topo_args(self, d: TopoCommonArgs) -> None:
        self.d = d

    def _add_qnode(self, name: str, d: NodeArgs | None = None) -> TopoQNode:
        if old_node := self.qnode_by_name.get(name):
            return old_node

        d = (self.d | d) if d else self.d
        mem_capacity = d.get("mem_capacity", -1)
        node: TopoQNode = {
            "name": name,
            "memory": {
                "capacity": max(0, mem_capacity),
                "t_cohere": d.get("t_cohere", 0.02),
                "time_decay": d.get("memory_decay"),
            },
        }
        self.qnodes.append(node)

        self.qnode_by_name[name] = node
        if mem_capacity < 0:
            self.extensible_memory_by_name[name] = node["memory"]
        return node

    def _inc_memory(self, name: str, n: int) -> None:
        if memory := self.extensible_memory_by_name.get(name):
            assert "capacity" in memory
            memory["capacity"] += n

    def _add_qchannel(
        self,
        node1: str,
        node2: str,
        d: ChannelArgs | None,
        length: float | None = None,
        capacity: int | tuple[int, int] | None = None,
    ):
        d = (self.d | d) if d else self.d

        caps: int | tuple[int, int] = capacity or d.get("ch_capacity", 1)
        cap1, cap2 = (caps, caps) if isinstance(caps, int) else caps

        la = d.get("link_arch", LinkArchDimBkSeq)
        la = LINK_ARCH_MAP.get(cast(LinkArchLiteral, la), cast(LinkArch | type[LinkArch], la))
        la = la() if callable(la) else la

        self._add_qnode(node1)
        self._add_qnode(node2)
        self._inc_memory(node1, cap1)
        self._inc_memory(node2, cap2)

        self.qchannels.append(
            {
                "node1": node1,
                "node2": node2,
                "capacity1": cap1,
                "capacity2": cap2,
                "parameters": {
                    "length": d.get("ch_length", 1.0) if length is None else length,
                    "link_arch": la,
                    "alpha": d.get("fiber_alpha", 0.2),
                    "transfer_error": d.get("fiber_error", "DEPOLAR:0.01"),
                    "bsa_error": d.get("bsa_error", "PERFECT"),
                },
            }
        )

    def topo(
        self,
        *,
        channels: Sequence[ChannelDef | tuple[NodePair, float] | tuple[NodePair, float, int | tuple[int, int]]],
        nodes: Sequence[NodeDef] = [],
        **kwargs: Unpack[TopoCommonArgs],
    ) -> Self:
        """
        Build a general topology.

        Args:
            channels: List of channels.
                Each element is either a ``ChannelDef`` or a tuple.
                If specified as tuple:
                * First tuple item is channel end points.
                * Second tuple item is channel length in kilometer.
                * Third tuple item is channel capacity, if it differs from ``kwargs.ch_capacity``.
            nodes: Node parameter overrides (optional).
                It is unnecessary to list a node unless its parameter differs from ``kwargs``.
            kwargs: Default node and channel parameters.
                To override a node's parameters, pass ``NodeDef`` to ``nodes``.
                To override a channel's parameters, pass ``ChannelDef`` to ``channels``.
        """
        self._save_topo_args(kwargs)

        for node in nodes:
            self._add_qnode(node.name, node.d)

        for ch in channels:
            capacity = None
            if isinstance(ch, ChannelDef):
                np = ch.np
                d = ch.d
                length = None
            else:
                np, length, *opt_capacity = ch
                if opt_capacity:
                    (capacity,) = opt_capacity
                d = None
            self._add_qchannel(*_split_node_pair(np), d, length, capacity)

        return self

    def topo_linear(
        self,
        *,
        nodes: int | Sequence[NodeDef | str],
        channels: Sequence[ChannelParam | float | tuple[float, int | tuple[int, int]]] | None = None,
        **kwargs: Unpack[TopoCommonArgs],
    ) -> Self:
        """
        Build a linear topology consisting of zero or more repeaters.

        Args:
            nodes: Number of nodes or list of node definitions / names, minimum is 2 nodes.
                If specified as number, the nodes are named ``S R1 R2 .. Rn D``.
                If specified as list, each node name must be unique.
            channels: Channel parameter overrides (optional).
                This is necessary only if some channel's parameter differs from ``kwargs``.
            kwargs: Default node and channel parameters.
                To override a node's parameters, pass ``NodeDef`` to ``nodes``.
                To override a channel's parameters, pass ``ChannelDef`` to ``channels``.
        """
        self._save_topo_args(kwargs)

        if isinstance(nodes, int):
            if nodes < 2:
                raise ValueError("at least two nodes")
            nodes = [f"R{i}" for i in range(nodes)]
            nodes[0] = "S"
            nodes[-1] = "D"

        n_nodes = len(nodes)
        if n_nodes < 2:
            raise ValueError("at least two nodes")
        n_links = n_nodes - 1
        if channels is None:
            chs = [None] * n_links
        elif len(channels) == n_links:
            chs = channels
        else:
            raise ValueError("incorrect number of channels")

        node_names: list[str] = []
        for node in nodes:
            if isinstance(node, NodeDef):
                self._add_qnode(node.name, node.d)
                node_names.append(node.name)
            else:
                self._add_qnode(node)
                node_names.append(node)

        for np, ch in zip(itertools.pairwise(node_names), chs, strict=True):
            if isinstance(ch, ChannelParam):
                self._add_qchannel(*np, ch.d)
            elif isinstance(ch, tuple):
                length, capacity = ch
                self._add_qchannel(*np, None, length, capacity)
            else:
                self._add_qchannel(*np, None, ch)

        return self

    def _assert_can_add_apps(self) -> None:
        if len(self.qnodes) == 0:
            raise TypeError("must define topology first")
        if len(self.qnode_apps) + len(self.controller_apps) > 0:
            raise TypeError("applications already installed")

    def _extract_apps_common_args(self, d: AppsCommonArgs) -> None:
        timing = d.pop("timing", None)
        if isinstance(timing, TimingMode):
            self.timing = timing
        elif timing is None:
            self.timing = TimingModeAsync()
        else:
            t_cohere = self.d.get("t_cohere", 0.02)
            if len(timing) == 0:
                timing = (t_cohere / 2 - 2 * CTRL_DELAY, 4 * CTRL_DELAY, t_cohere / 2 - 2 * CTRL_DELAY)
            self.timing = TimingModeSync(durations=timing)

    def _add_link_layer(self):
        self.qnode_apps.append(
            LinkLayer(
                attempt_rate=self.d.get("entg_attempt_rate", 50e6),
                init_fidelity=self.d.get("init_fidelity", 0.99),
                eta_d=self.d.get("eta_d", 0.95),
                eta_s=self.d.get("eta_s", 0.95),
                frequency=self.d.get("frequency", 1e6),
            )
        )

    def proactive_centralized(
        self,
        **kwargs: Unpack[AppsForwarderArgs],
    ) -> Self:
        """
        Choose proactive forwarding with centralized control.

        Args:
            mux: Multiplexing scheme, default is buffer-space.
        """
        self._assert_can_add_apps()
        self._extract_apps_common_args(kwargs)
        kwargs.setdefault("p_swap", 0.5)

        mux = kwargs.get("mux")
        if mux is None or isinstance(mux, MuxSchemeBufferSpace):
            self.qubit_allocation = QubitAllocationType.FOLLOW_QCHANNEL
        elif isinstance(self.route, YenRouteAlgorithm):
            raise TypeError("YenRouteAlgorithm is only compatible with MuxSchemeBufferSpace")

        self._add_link_layer()
        self.qnode_apps.append(
            ProactiveForwarder(**kwargs),
        )
        self.controller_apps.append(
            ProactiveRoutingController(),
        )
        return self

    def proactive_distributed(self) -> Self:
        self._assert_can_add_apps()
        raise NotImplementedError

    def reactive_centralized(
        self,
        *,
        swap: SwapPolicy = "asap",
        **kwargs: Unpack[AppsForwarderArgs],
    ) -> Self:
        """
        Choose reactive forwarding with centralized control.

        Args:
            mux: Multiplexing scheme, default is buffer-space.
            swap: SwapPolicy for routes.

        ``.request()`` method only accepts src-dst nodes, but does not support ``RoutingPath``.
        """
        self._assert_can_add_apps()
        self._extract_apps_common_args(kwargs)
        kwargs.setdefault("p_swap", 0.5)

        self._add_link_layer()
        self.qnode_apps.append(
            ReactiveForwarder(**kwargs),
        )
        self.controller_apps.append(
            ReactiveRoutingController(swap=swap),
        )
        return self

    def reactive_distributed(self) -> Self:
        self._assert_can_add_apps()
        raise NotImplementedError

    def external_controller(
        self,
        *,
        nats_prefix=ClassicBridge.DEFAULT_NATS_PREFIX,
    ) -> Self:
        """
        Replace the controller application with ``ClassicBridge``.

        Args:
            nats_prefix: Prefix of NATS subjects.

        This must be called after ``proactive_centralized`` or ``reactive_centralized``.
        The internal controller application is deleted and replaced with ``ClassicBridge``, which allows
        the controller logic to be implemented in an external program connected over NATS.

        ``.request()`` method cannot be used.
        Instead, requests or routing paths should be defined in the external controller.
        """
        self.controller_apps.clear()
        self.controller_apps.append(ClassicBridge(nats_prefix=nats_prefix))
        return self

    def _to_path(self, arg1: RoutingPath | NodePair, d: RoutingPathInitArgs) -> RoutingPath:
        if isinstance(arg1, RoutingPath):
            return arg1
        if isinstance(self.route, YenRouteAlgorithm):
            return RoutingPathMulti(*_split_node_pair(arg1), **d)
        return RoutingPathSingle(*_split_node_pair(arg1), **d, qubit_allocation=self.qubit_allocation)

    @functools.singledispatchmethod
    def _add_request(self, ctrl: Application, arg1: RoutingPath | NodePair, d: RoutingPathInitArgs) -> None:
        _ = arg1, d
        raise NotImplementedError(f"{type(ctrl)} does not support .request() method")

    @_add_request.register
    def _(self, ctrl: ProactiveRoutingController, arg1: RoutingPath | NodePair, d: RoutingPathInitArgs) -> None:
        ctrl.paths.append(self._to_path(arg1, d))

    @_add_request.register
    def _(self, ctrl: ReactiveRoutingController, arg1: RoutingPath | NodePair, d: RoutingPathInitArgs) -> None:
        _ = d
        if isinstance(arg1, RoutingPath):
            raise TypeError(f"{type(ctrl)} does not support .request(RoutingPath)")
        self.requests.append(_split_node_pair(arg1))

    @overload
    def request(self, src_dst: NodePair, /, **kwargs: Unpack[RoutingPathInitArgs]) -> Self:
        """
        Define a request that may use one or more paths determined by routing algorithm.
        """

    @overload
    def request(self, rp: RoutingPath, /) -> Self:
        """
        Define a request that is constrained to a specific path.
        """

    def request(
        self,
        arg1: RoutingPath | NodePair,
        /,
        **kwargs: Unpack[RoutingPathInitArgs],
    ) -> Self:
        if len(self.controller_apps) == 0:
            raise TypeError("must install controller application first")
        self._add_request(self.controller_apps[0], arg1, kwargs)
        return self

    def make_topo(self) -> Topology:
        """
        Retrieve topology object.

        This method is only necessary if you need to inspect or modify the topology factory object.
        Otherwise, use ``.make_network()`` directly.
        """
        topo = Topo(qnodes=self.qnodes, qchannels=self.qchannels)
        if len(self.controller_apps) > 0:
            topo["controller"] = TopoController(name="ctrl", apps=self.controller_apps)
        return CustomTopology(
            topo,
            nodes_apps=self.qnode_apps,
        )

    def make_network(
        self,
        *,
        topo: Topology | None = None,
        connect_controller=True,
    ) -> QuantumNetwork:
        """
        Construct quantum network.

        Args:
            topo: Result of ``.make_topo()`` method with possible modification, defaults to ``self.make_topo()``.
            connect_controller: If True and controller exists, create cchannels between controller and each qnode.

        Returns: QuantumNetwork ready for simulation.
        """
        topo = topo or self.make_topo()
        net = QuantumNetwork(
            topo,
            classic_topo=ClassicTopology.Follow,
            route=self.route,
            timing=self.timing,
            epr_type=self.epr_type,
        )
        for src, dst in self.requests:
            net.add_request(net.get_node(src), net.get_node(dst))

        if connect_controller and topo.controller:
            topo.connect_controller(net.nodes, delay=CTRL_DELAY)

        return net
