import functools
from collections.abc import Mapping, Sequence
from typing import Literal, Self, TypedDict, Unpack, cast, overload

from tap import Tap

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


class TopoCommonArgs(TypedDict, total=False):
    t_cohere: float
    """Memory coherence time in seconds, defaults to ``0.02``."""
    memory_decay: TimeDecayInput
    """Memory time decay function, defaults to dephasing in ``t_cohere``."""
    entg_attempt_rate: float
    """Maximum entanglement attempts per second, defaults to ``50_000_000`` but currently ineffective."""
    init_fidelity: float | None
    """
    Fidelity of generated entangled pairs, defaults to ``0.99``.
    If ``None``, determine with error models in link architecture.
    """
    eta_d: float
    """Detector efficiency, defaults to ``0.95``."""
    eta_s: float
    """Source efficiency, defaults to ``0.95``."""
    frequency: float
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


def _broadcast[T](name: str, input: T | Sequence[T], n: int) -> Sequence[T]:
    if isinstance(input, Sequence) and not isinstance(input, (str, bytes, bytearray, memoryview)):
        if len(input) != n:
            raise ValueError(f"{name} must have {n} items")
        return input
    return [cast(T, input)] * n


def _split_node_pair(np: NodePair) -> tuple[str, str]:
    if isinstance(np, str):
        tokens = np.split("-")
        if len(tokens) != 2:
            raise ValueError(f"expect two node names in '{np}'")
        return cast(tuple[str, str], tuple(tokens))
    return np


def _split_channel_capacity(item: int | tuple[int, int]) -> tuple[int, int]:
    return (item, item) if isinstance(item, int) else item


def _convert_link_arch(la: LinkArchDef) -> LinkArch:
    la = LINK_ARCH_MAP.get(cast(LinkArchLiteral, la), cast(LinkArch | type[LinkArch], la))
    return la() if callable(la) else la


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
        self.qnode_apps: list[Application] = []
        self.qchannels: list[TopoQChannel] = []
        self.controller_apps: list[Application] = []

        self.qubit_allocation = QubitAllocationType.DISABLED
        self.requests: list[tuple[str, str]] = []

    def _parse_topo_args(self, d: TopoCommonArgs) -> None:
        self.t_cohere = d.get("t_cohere", 0.02)
        self.memory_decay = d.get("memory_decay")
        self.entg_attempt_rate = d.get("entg_attempt_rate", 50e6)
        self.init_fidelity = d.get("init_fidelity", 0.99)
        self.eta_d = d.get("eta_d", 0.95)
        self.eta_s = d.get("eta_s", 0.95)
        self.frequency = d.get("frequency", 1e6)

    def _add_qnode(self, name: str, mem_cap: int) -> TopoQNode:
        node: TopoQNode = {
            "name": name,
            "memory": {"capacity": mem_cap},
        }
        self.qnodes.append(node)
        return node

    def _add_qchannel(
        self,
        node1: str,
        node2: str,
        cap1: int,
        cap2: int,
        length: float,
        fiber_alpha: float,
        fiber_error: ErrorModelInputLength,
        bsa_error: ErrorModelInputBasic,
        la: LinkArchDef,
    ):
        self.qchannels.append(
            {
                "node1": node1,
                "node2": node2,
                "capacity1": cap1,
                "capacity2": cap2,
                "parameters": {
                    "length": length,
                    "link_arch": _convert_link_arch(la),
                    "alpha": fiber_alpha,
                    "transfer_error": fiber_error,
                    "bsa_error": bsa_error,
                },
            }
        )

    def topo(
        self,
        *,
        mem_capacity: int | Mapping[str, int] | None = None,
        channels: Sequence[tuple[NodePair, float] | tuple[NodePair, float, int | tuple[int, int]]],
        channel_capacity: int = 1,
        fiber_alpha: float = 0.2,
        fiber_error: ErrorModelInputLength = "DEPOLAR:0.01",
        bsa_error: ErrorModelInputBasic = "PERFECT",
        link_arch: LinkArchDef | Sequence[LinkArchDef] = LinkArchDimBkSeq,
        **kwargs: Unpack[TopoCommonArgs],
    ) -> Self:
        """
        Build a general topology.

        Args:
            mem_capacity: Number of memory qubits per node.
                If specified as number, it is applied to all nodes.
                If specified as dict, it maps from node name to node memory capacity.
                If ``None`` or for unspecified nodes, it is derived from channels.
            channels: List of channels.
                First tuple item is channel end points; nodes are automatically created.
                Second tuple item is channel length in kilometer.
                Third tuple item is channel capacity, defaults to ``channel_capacity``;
                for each qchannel, an integer applies to both sides, a tuple applies to (left,right) sides.
            channel_capacity: Qubit allocation per qchannel, if not specified in ``channels``.
            fiber_alpha: Fiber loss in dB/km, determines success probability.
            fiber_error: Fiber error model, determines qualify of entangled state.
            bsa_error: Photonic Bell-state analyzer or absorptive memory capture error model,
                       determines qualify of entangled state.
            link_arch: Link architecture per qchannel.
                If specified as list, it must have same length as ``channel_length``.
                If specified as instance/type/string, it is broadcast to all qchannels.
        """
        self._parse_topo_args(kwargs)

        n_links = len(channels)
        link_arch = _broadcast("link_arch", link_arch, n_links)
        node_by_name: dict[str, TopoQNode | Literal[True]] = {}

        def ensure_node(name: str, ch_cap: int):
            node = node_by_name.get(name)
            if node is True:  # node created with mem_capacity
                return

            if node is not None:  # need to increase mem_capacity
                assert "memory" in node
                assert "capacity" in node["memory"]
                node["memory"]["capacity"] += ch_cap
                return

            # need to create new node
            if mem_capacity is None:
                mem_cap, defined_mem_cap = ch_cap, False
            elif isinstance(mem_capacity, int):
                mem_cap, defined_mem_cap = mem_capacity, True
            elif name in mem_capacity:
                mem_cap, defined_mem_cap = mem_capacity[name], True
            else:
                mem_cap, defined_mem_cap = ch_cap, False

            node = self._add_qnode(name, mem_cap)
            node_by_name[name] = True if defined_mem_cap else node

        for (np, length, *opt_cap), la in zip(channels, link_arch, strict=True):
            node1, node2 = _split_node_pair(np)
            cap1, cap2 = _split_channel_capacity(opt_cap[0] if len(opt_cap) > 0 else channel_capacity)
            ensure_node(node1, cap1)
            ensure_node(node2, cap2)
            self._add_qchannel(node1, node2, cap1, cap2, length, fiber_alpha, fiber_error, bsa_error, la)

        return self

    def topo_linear(
        self,
        *,
        nodes: int | Sequence[str],
        mem_capacity: int | Sequence[int] | None = None,
        channel_length: float | Sequence[float],
        channel_capacity: int | Sequence[int | tuple[int, int]] = 1,
        fiber_alpha: float = 0.2,
        fiber_error: ErrorModelInputLength = "DEPOLAR:0.01",
        bsa_error: ErrorModelInputBasic = "PERFECT",
        link_arch: LinkArchDef | Sequence[LinkArchDef] = LinkArchDimBkSeq,
        **kwargs: Unpack[TopoCommonArgs],
    ) -> Self:
        """
        Build a linear topology consisting of zero or more repeaters.

        Args:
            nodes: Number of nodes or list of node names, minimum is 2.
                If specified as number, the nodes are named ``S R1 R2 .. Rn D``.
                If specified as list, each node name must be unique.
            mem_capacity: Number of memory qubits per node.
                If specified as list, it must have same length as ``nodes``.
                If specified as number, it is broadcast to all nodes.
                If ``None``, it is derived from ``channel_capacity``.
            channel_length: Lengths of qchannels between adjacent nodes in kilometer.
                If specified as list, it must have length ``len(nodes)-1``.
                If specified as number, all qchannels have uniform length.
            channel_capacity: Qubit allocation per qchannel.
                If specified as list, it must have same length as ``channel_length``.
                If specified as number, it is broadcast to all qchannels.
                For each qchannel, an integer applies to both sides, a tuple applies to (left,right) sides.
            fiber_alpha: Fiber loss in dB/km, determines success probability.
            fiber_error: Fiber error model, determines qualify of entangled state.
            bsa_error: Photonic Bell-state analyzer or absorptive memory capture error model,
                       determines qualify of entangled state.
            link_arch: Link architecture per qchannel.
                If specified as list, it must have same length as ``channel_length``.
                If specified as instance/type/string, it is broadcast to all qchannels.
        """
        self._parse_topo_args(kwargs)

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

        channel_length = _broadcast("channel_length", channel_length, n_links)
        channel_capacity = _broadcast("channel_capacity", channel_capacity, n_links)
        link_arch = _broadcast("link_arch", link_arch, n_links)

        if mem_capacity is None:
            mem_capacity = [0]
            for caps in channel_capacity:
                capL, capR = _split_channel_capacity(caps)
                mem_capacity[-1] += capL
                mem_capacity.append(capR)
        else:
            mem_capacity = _broadcast("mem_capacity", mem_capacity, n_nodes)

        for name, mem_capacity in zip(nodes, mem_capacity, strict=True):
            self._add_qnode(name, mem_capacity)

        for i, (length, caps, la) in enumerate(zip(channel_length, channel_capacity, link_arch, strict=True)):
            node1, node2 = nodes[i], nodes[i + 1]
            cap1, cap2 = _split_channel_capacity(caps)
            self._add_qchannel(node1, node2, cap1, cap2, length, fiber_alpha, fiber_error, bsa_error, la)

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
            if len(timing) == 0:
                timing = (self.t_cohere / 2 - 2 * CTRL_DELAY, 4 * CTRL_DELAY, self.t_cohere / 2 - 2 * CTRL_DELAY)
            self.timing = TimingModeSync(durations=timing)

    def _add_link_layer(self):
        self.qnode_apps.append(
            LinkLayer(
                attempt_rate=self.entg_attempt_rate,
                init_fidelity=self.init_fidelity,
                eta_d=self.eta_d,
                eta_s=self.eta_s,
                frequency=self.frequency,
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
            memory_args={
                "t_cohere": self.t_cohere,
                "time_decay": self.memory_decay,
            },
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
