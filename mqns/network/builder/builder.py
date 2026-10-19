import itertools
from collections.abc import Callable, Sequence
from typing import Literal, Self, TypedDict, Unpack, cast, overload

from mqns.entity.memory import QuantumMemoryInitKwargs
from mqns.entity.node import Application, NodePair, QNode, split_node_pair
from mqns.entity.qchannel import LinkArchInput, QuantumChannel
from mqns.models.epr import EprTypeInput
from mqns.models.error import TimeDecayInput
from mqns.models.error.input import ErrorModelInputBasic, ErrorModelInputLength
from mqns.network.builder.topogen import TopoGen
from mqns.network.fw import ForwarderInitKwargs, RoutingPath, RoutingPathInitArgs
from mqns.network.network import QuantumNetwork, Request, TimingMode, TimingModeAsync, TimingModeSync
from mqns.network.proactive import MuxSchemeInput, ProactiveForwarder, ProactiveRoutingController
from mqns.network.protocol.classicbridge import ClassicBridge
from mqns.network.protocol.consumer import Consumer
from mqns.network.protocol.link_layer import LinkLayer, LinkLayerInitKwargs
from mqns.network.reactive import ReactiveForwarder, ReactiveRoutingController
from mqns.network.route import DijkstraRouteAlgorithm, RouteAlgorithm
from mqns.network.topology import ClassicTopology
from mqns.network.topology.customtopo import CustomTopology, Topo, TopoController, TopoQChannel, TopoQNode

CTRL_DELAY = 5e-06
"""
Delay of the classic channels between the controller and each QNode, in seconds.

In most examples, the overall simulation duration is increased by this value,
so that the QNodes can perform entanglement forwarding for the full intended duration.
"""


def _name_nodes(n: int, nodes_naming: Literal["n1", "A", "R1", "R"]) -> Sequence[str]:
    if nodes_naming == "A":
        if n > 26:
            raise ValueError("too many nodes for nodes_naming='A'")
        return [chr(0x41 + i) for i in range(n)]
    elif nodes_naming == "n1":
        return [f"n{i}" for i in range(1, 1 + n)]
    elif nodes_naming == "R" and n == 3:
        return ["S", "R", "D"]
    else:
        return ["S", *(f"R{i}" for i in range(1, n - 1)), "D"]


class NodeArgs(TypedDict, total=False):
    """Node parameters."""

    mem_capacity: int
    """Memory capacity, defaults to ``-1`` for deriving from channel capacities."""
    t_cohere: float
    """Memory coherence time in seconds, defaults to ``0.02``."""
    memory_decay: TimeDecayInput
    """Memory time decay function, defaults to dephasing in ``t_cohere``."""
    frequency: float
    """Entanglement source frequency, defaults to ``1_000_000``."""
    tau_0: float
    """Local operation delay in seconds for emitting and absorbing photon, defaults to ``0.0``."""


class NodeDef:
    """Node definition."""

    def __init__(self, name: str, **kwargs: Unpack[NodeArgs]):
        self.name = name
        self.d = kwargs


class ChannelArgs(TypedDict, total=False):
    """Channel parameters."""

    ch_length: float
    """
    Channel length in kilometer, defaults to ``1.0``.
    """
    ch_capacity: int | tuple[int, int]
    """
    Channel capacity, defaults to ``1``.
    An integer applies to both sides; a tuple applies to ``(left,right)`` sides.
    """
    link_arch: LinkArchInput
    """
    Link architecture, defaults to ``LinkArchDimBkSeq``.
    """
    fiber_alpha: float
    """
    Fiber loss in dB/km, defaults to ``0.2``.
    This determines success probability.
    """
    eta_s: float
    """
    Source efficiency between 0 and 1, defaults to ``0.95``.
    This determines success probability.
    """
    eta_d: float
    """
    Detector efficiency between 0 and 1, defaults to ``0.95``.
    This determines success probability.
    """
    init_fidelity: float | Sequence[float] | None
    """
    Fidelity of generated entangled pairs, defaults to ``0.99``.
    If ``None``, determine with error models in link architecture.
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


def _channel_args_from_tuple(input: Sequence) -> ChannelArgs:
    """
    Parse a channel args tuple.

    The input could be one of:

    * ``tuple[()]``: none.
    * ``tuple[float]``: channel length.
    * ``tuple[float, int | tuple[int,int]]``: channel length and capacity.
    """
    a = ChannelArgs()
    if len(input) >= 1:
        a["ch_length"] = input[0]
    if len(input) >= 2:
        a["ch_capacity"] = input[1]
    return a


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
    Combination of ``NodeArgs`` and ``ChannelArgs``.
    """


class AppsCommonArgs(TypedDict, total=False):
    timing: TimingMode | Sequence[float] | None
    """
    Network timing mode, defaults to ASYNC.
    If specified as three floats, construct ``TimingModeSync`` with these durations.
    """

    ll: LinkLayerInitKwargs
    """
    LinkLayer parameters.
    """


class AppsForwarderArgs(AppsCommonArgs, ForwarderInitKwargs):
    """
    Combination of ``AppsCommonArgs`` and ``ForwarderInitKwargs``.

    Args:
        p_swap: Probability of successful entanglement swapping in forwarder, defaults to ``0.5``.
    """


class NetworkBuilder:
    """
    Orchestrator of quantum network simulation with :py:class:`~mqns.network.protocol.link_layer.LinkLayer`
    and :py:class:`~mqns.network.fw.Forwarder`.

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
        epr_type: EprTypeInput = "W",
    ):
        """
        Constructor.

        Args:
            route: Route algorithm, defaults to Dijkstra.
            epr_type: Network-wide EPR model, defaults to Werner state.
        """

        self.route = route
        self.epr_type = epr_type

        self.qnodes: list[TopoQNode] = []
        self.qnode_by_name: dict[str, TopoQNode] = {}
        self.extensible_memory_by_node: dict[str, QuantumMemoryInitKwargs] = {}
        self.link_layer_by_node: dict[str, LinkLayerInitKwargs] = {}
        self.qnode_apps: list[Application] = []
        self.qchannels: list[TopoQChannel] = []
        self.controller_apps: list[Application] = []

        self.requests: list[Request] = []

    def _save_topo_args(self, d: TopoCommonArgs) -> None:
        self.d = d

    def _add_qnode(self, name: str, d: NodeArgs) -> TopoQNode:
        if old_node := self.qnode_by_name.get(name):
            return old_node

        d = self.d | d
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
            self.extensible_memory_by_node[name] = node["memory"]

        self.link_layer_by_node[name] = LinkLayerInitKwargs(
            frequency=d.get("frequency", 1e6),
            tau_0=d.get("tau_0", 0.0),
        )

        return node

    def _inc_memory(self, name: str, n: int) -> None:
        if memory := self.extensible_memory_by_node.get(name):
            assert "capacity" in memory
            memory["capacity"] += n

    def _add_qchannel(
        self,
        node1: str,
        node2: str,
        d: ChannelArgs,
    ):
        d = self.d | d
        caps: int | tuple[int, int] = d.get("ch_capacity", 1)
        cap1, cap2 = (caps, caps) if isinstance(caps, int) else caps

        self._add_qnode(node1, {})
        self._add_qnode(node2, {})
        self._inc_memory(node1, cap1)
        self._inc_memory(node2, cap2)

        self.qchannels.append(
            {
                "node1": node1,
                "node2": node2,
                "capacity1": cap1,
                "capacity2": cap2,
                "parameters": {
                    "length": d.get("ch_length", 1.0),
                    "link_arch": d.get("link_arch"),
                    "alpha": d.get("fiber_alpha", 0.2),
                    "eta_s": d.get("eta_s", 0.95),
                    "eta_d": d.get("eta_d", 0.95),
                    "init_fidelity": d.get("init_fidelity", 0.99),
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
                Each element is either a :py:class:`ChannelDef` or a tuple.
                If specified as tuple:

                * First tuple item is channel end points.
                * Second tuple item is channel length in kilometer.
                * Third tuple item is channel capacity, if it differs from ``kwargs.ch_capacity``.

            nodes: Node parameter overrides (optional).
                It is unnecessary to list a node unless its parameter differs from ``kwargs``.
            kwargs: Default node and channel parameters.
                To override a node's parameters, pass :py:class:`NodeDef` to ``nodes``.
                To override a channel's parameters, pass :py:class:`ChannelDef` to ``channels``.
        """
        self._save_topo_args(kwargs)

        for node in nodes:
            self._add_qnode(node.name, node.d)

        for ch in channels:
            if isinstance(ch, ChannelDef):
                self._add_qchannel(*split_node_pair(ch.np), ch.d)
            else:
                np, *length_capacity = ch
                self._add_qchannel(*split_node_pair(np), _channel_args_from_tuple(length_capacity))

        return self

    def topo_from(
        self,
        topo: TopoGen,
        *,
        nodes_naming: Literal["n1", "A", "R1", "R"] = "n1",
        nodes: Sequence[NodeDef] = [],
        channels: Sequence[ChannelDef | tuple[NodePair, float] | tuple[NodePair, float, int | tuple[int, int]]] = [],
        **kwargs: Unpack[TopoCommonArgs],
    ) -> Self:
        """
        Build a topology with a topology generator.

        Args:
            nodes_naming: Node name pattern.

                * *n1*: ``n1 n2 ..``.
                * *A*: ``A B ..``, limited to 26 nodes.
                * *R1*: ``S R1 R2 .. Rn D``; only suitable for linear topology.
                * *R*: ``S R D`` for three nodes, otherwise same as *R1*; only suitable for linear topology.

            nodes: Node parameter overrides (optional).
                It is unnecessary to list a node unless its parameter differs from ``kwargs``.
            channels: Channel parameter overrides (optional).
                It is unnecessary to list a channel unless its parameter differs from ``kwargs``.
        """
        self._save_topo_args(kwargs)

        for node in nodes:
            self._add_qnode(node.name, node.d)

        ch_defs: dict[tuple[str, str], ChannelArgs] = {}
        for ch in channels:
            if isinstance(ch, tuple):
                ch_defs[split_node_pair(ch[0])] = _channel_args_from_tuple(ch[1:])
            else:
                ch_defs[split_node_pair(ch.np)] = ch.d
        node_names: Sequence[str] = []

        def make_nodes(n: int) -> Sequence[str]:
            nonlocal node_names
            node_names = _name_nodes(n, nodes_naming)
            for name in node_names:
                self._add_qnode(name, {})
            return node_names

        def make_link(a: int | str, b: int | str) -> None:
            a = node_names[a] if type(a) is int else cast(str, a)
            b = node_names[b] if type(b) is int else cast(str, b)
            d = ch_defs.get((a, b)) or ch_defs.get((b, a), {})
            self._add_qchannel(a, b, d)

        topo.build_topo(make_nodes=make_nodes, make_link=make_link)
        return self

    def topo_linear(
        self,
        *,
        nodes: int | Sequence[NodeDef | str],
        nodes_naming: Literal["n1", "A", "R1", "R"] = "R",
        channels: Sequence[ChannelParam | tuple[()] | float | tuple[float, int | tuple[int, int]]] | None = None,
        **kwargs: Unpack[TopoCommonArgs],
    ) -> Self:
        """
        Build a linear topology consisting of zero or more repeaters.

        Args:
            nodes: Number of nodes or list of node definitions / names, minimum is 2 nodes.
                If specified as number, the nodes are named according to ``nodes_naming``.
                If specified as list, each node name must be unique.
            nodes_naming: Node name pattern, used if ``nodes`` is an integer.
                See :py:meth:`topo_from` for naming rules.
            channels: Channel parameter overrides (optional).
                This is necessary only if some channel's parameter differs from ``kwargs``.
            kwargs: Default node and channel parameters.
                To override a node's parameters, pass :py:class:`NodeDef` to ``nodes``.
                To override a channel's parameters, pass :py:class:`ChannelDef` to ``channels``.
        """
        self._save_topo_args(kwargs)

        if isinstance(nodes, int):
            if nodes < 2:
                raise ValueError("at least two nodes")
            n_nodes = nodes
            nodes = _name_nodes(n_nodes, nodes_naming)
        else:
            n_nodes = len(nodes)
            if n_nodes < 2:
                raise ValueError("at least two nodes")

        n_links = n_nodes - 1
        if channels is None:
            chs = [()] * n_links
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
                self._add_qnode(node, {})
                node_names.append(node)

        for np, ch in zip(itertools.pairwise(node_names), chs, strict=True):
            if isinstance(ch, ChannelParam):
                self._add_qchannel(*np, ch.d)
            elif isinstance(ch, tuple):
                self._add_qchannel(*np, _channel_args_from_tuple(ch))
            else:
                self._add_qchannel(*np, {"ch_length": ch})

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

        self.link_layer_args = d.pop("ll", {})

    def _add_link_layer(self):
        for name, node in self.qnode_by_name.items():
            node.setdefault("apps", []).append(LinkLayer(**(self.link_layer_by_node[name] | self.link_layer_args)))

    def _add_consumer(self):
        self.qnode_apps.append(Consumer())

    def proactive_centralized(
        self,
        *,
        mux: MuxSchemeInput = None,
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

        self._add_link_layer()
        self.qnode_apps.append(
            ProactiveForwarder(mux=mux, **kwargs),
        )
        self._add_consumer()
        self.controller_apps.append(
            ProactiveRoutingController(mux=mux),
        )
        return self

    def proactive_distributed(self) -> Self:
        """
        Choose proactive forwarding with distributed control.

        This is unimplemented.
        """
        self._assert_can_add_apps()
        raise NotImplementedError

    def reactive_centralized(self, **kwargs: Unpack[AppsForwarderArgs]) -> Self:
        """
        Choose reactive forwarding with centralized control.
        """
        self._assert_can_add_apps()
        self._extract_apps_common_args(kwargs)
        kwargs.setdefault("p_swap", 0.5)

        self._add_link_layer()
        self.qnode_apps.append(
            ReactiveForwarder(**kwargs),
        )
        self._add_consumer()
        self.controller_apps.append(
            ReactiveRoutingController(),
        )
        return self

    def reactive_distributed(self) -> Self:
        """
        Choose reactive forwarding with distributed control.

        This is unimplemented.
        """
        self._assert_can_add_apps()
        raise NotImplementedError

    def external_controller(
        self,
        *,
        nats_prefix=ClassicBridge.DEFAULT_NATS_PREFIX,
    ) -> Self:
        """
        Replace the controller application with :py:class:`mqns.network.protocol.classicbridge.ClassicBridge`.

        Args:
            nats_prefix: Prefix of NATS subjects.

        This must be called after :py:meth:`proactive_centralized` or :py:meth:`reactive_centralized`.
        The internal controller application is deleted and replaced with
        :py:class:`~mqns.network.protocol.classicbridge.ClassicBridge`, which allows
        the controller logic to be implemented in an external program connected over NATS.

        :py:meth:`request` method cannot be used.
        Instead, requests or routing paths should be defined in the external controller.
        """
        self.controller_apps.clear()
        self.controller_apps.append(ClassicBridge(nats_prefix=nats_prefix))
        return self

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
            raise TypeError("must install controller application")

        if isinstance(arg1, RoutingPath):
            self.requests.append(Request(arg1))
        else:
            self.requests.append(Request(arg1).path(**kwargs))

        return self

    def make_network(
        self,
        *,
        modify: Callable[[Self, Topo], None] | None = None,
        connect_controller=True,
    ) -> QuantumNetwork:
        """
        Construct quantum network.

        Args:
            modify: Callback function to modify the topology definition if necessary.
            connect_controller: If True and controller exists, create cchannels between controller and each qnode.

        Returns:
            QuantumNetwork ready for simulation.
        """
        if len(self.qnode_apps) == 0:
            raise TypeError("must install applications")
        topo_def = Topo(qnodes=self.qnodes, qchannels=self.qchannels)
        if len(self.controller_apps) > 0:
            topo_def["controller"] = TopoController(name="ctrl", apps=self.controller_apps)
        if modify:
            modify(self, topo_def)

        topo = CustomTopology(topo_def, nodes_apps=self.qnode_apps)
        net = QuantumNetwork(
            topo,
            classic_topo=ClassicTopology.Follow,
            route=self.route,
            timing=self.timing,
            epr_type=self.epr_type,
        )
        net.add_request(*self.requests)

        if connect_controller and topo.controller:
            topo.connect_controller(net.nodes, delay=CTRL_DELAY)

        return net
