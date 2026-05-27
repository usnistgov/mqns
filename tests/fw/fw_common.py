import copy
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Literal, TypedDict, Unpack, override

import pytest

from mqns.entity.cchannel import ClassicChannelInitKwargs, ClassicPacket
from mqns.entity.memory import QubitState
from mqns.entity.node import Application, Controller, Node, QNode
from mqns.entity.qchannel import LinkArchAlways, LinkArchDimBk, QuantumChannelInitKwargs
from mqns.models.epr import Entanglement, WernerStateEntanglement
from mqns.network.fw import Forwarder, ForwarderInitKwargs, RoutingController, RoutingPath
from mqns.network.fw.fw_swap import ForwarderSwapProc
from mqns.network.network import QuantumNetwork, TimingMode, TimingModeAsync
from mqns.network.proactive import ProactiveForwarder, ProactiveRoutingController
from mqns.network.protocol.event import QubitEntangledEvent, QubitReleasedEvent
from mqns.network.protocol.link_layer import LinkLayer
from mqns.network.reactive import ReactiveForwarder, ReactiveRoutingController
from mqns.network.route import RouteAlgorithm, YenRouteAlgorithm
from mqns.network.topology import ClassicTopology, GridTopology, LinearTopology, Topology, TopologyInitKwargs, TreeTopology
from mqns.simulator import Event, Simulator, Time, func_to_event
from mqns.utils import AutoIncrementIdentifier, log


class QubitReleaseLoggerApp(Application):
    """Record when qubits are released."""

    def __init__(self):
        super().__init__()
        self.history: list[tuple[int, Time]] = []
        """Qubit address and release time."""

    @override
    def handle(self, event: Event) -> bool | None:
        if type(event) is QubitReleasedEvent:
            self.history.append((event.qubit.addr, event.t))
            log.debug(f"{self}: RELEASE {event.qubit}")
        return False

    @property
    def last_time(self) -> Time:
        """Retrieve the timestamp of last release event."""
        if not self.history:
            return Time.SENTINEL
        return self.history[-1][1]


dflt_qchannel_args = QuantumChannelInitKwargs(
    length=100,  # delay is 0.0005 seconds
    link_arch=LinkArchAlways(LinkArchDimBk()),  # etg creation in 0.001 seconds and arrival in 0.002 seconds
)

dflt_cchannel_args = ClassicChannelInitKwargs(
    length=100,  # delay is 0.0005 seconds
)


class BuildNetworkArgs(TypedDict, total=False):
    mode: Literal["P", "R"]
    t_cohere: float  # memory dephasing time, defaults to 5.0 seconds
    qchannel_capacity: int  # quantum channel capacity, defaults to 1
    qchannel_args: QuantumChannelInitKwargs
    cchannel_args: ClassicChannelInitKwargs
    ctrl: RoutingController  # replacing controller application
    fw: ForwarderInitKwargs  # forwarder parameters (`p_swap` defaults to 0.5)
    swap_table_leak_tol: int  # ForwarderSwapProc memory leak tolerance
    end_time: float  # simulation end time, defaults to 10.0 seconds
    timing: TimingMode  # network timing mode, defaults to ASYNC
    epr_type: type[Entanglement]  # entanglement type, defaults to werner state
    has_link_layer: bool  # whether to include full LinkLayer application, defaults to False
    init_fidelity: float  # initial fidelity, defaults to 0.99


def _make_topo_args(d: BuildNetworkArgs, *, memory_capacity_factor: int) -> TopologyInitKwargs:
    qchannel_capacity = d.get("qchannel_capacity", 1)

    nodes_apps: list[Application[QNode]] = []
    if d.get("has_link_layer", False):
        nodes_apps.append(LinkLayer(init_fidelity=d.get("init_fidelity", 0.99)))
    else:
        nodes_apps.append(QubitReleaseLoggerApp())

    fw_args = copy.copy(d.get("fw")) or {}
    fw_args.setdefault("p_swap", 0.6)
    match d.get("mode", "P"):
        case "P":
            nodes_apps.append(ProactiveForwarder(**fw_args))
        case "R":
            nodes_apps.append(ReactiveForwarder(**fw_args))

    return TopologyInitKwargs(
        nodes_apps=nodes_apps,
        qchannel_args=d.get("qchannel_args", dflt_qchannel_args),
        cchannel_args=d.get("cchannel_args", dflt_cchannel_args),
        memory_args={
            "t_cohere": d.get("t_cohere", 5.0),
            "capacity": memory_capacity_factor * qchannel_capacity,
        },
    )


def _build_network_finish(
    topo: Topology,
    d: BuildNetworkArgs,
    *,
    route: RouteAlgorithm | None = None,
):
    ForwarderSwapProc.table_leak_tol = d.get("swap_table_leak_tol", 0)

    qchannel_capacity = d.get("qchannel_capacity", 1)

    if (ctrl := d.get("ctrl")) is None:
        match d.get("mode", "P"):
            case "P":
                ctrl = ProactiveRoutingController()
            case "R":
                ctrl = ReactiveRoutingController()
    topo.controller = Controller("ctrl", apps=[ctrl])

    net = QuantumNetwork(
        topo=topo,
        classic_topo=ClassicTopology.Follow,
        route=route,
        timing=d.get("timing", TimingModeAsync()),
        epr_type=d.get("epr_type", WernerStateEntanglement),
    )
    for qchannel in net.qchannels:
        qchannel.assign_memory_qubits(capacity=qchannel_capacity)
    topo.connect_controller(net.nodes)

    simulator = Simulator(0.0, d.get("end_time", 10.0), install_to=(log, net))

    return net, simulator


def build_linear_network(
    n_nodes: int,
    **kwargs: Unpack[BuildNetworkArgs],
) -> tuple[QuantumNetwork, Simulator]:
    topo = LinearTopology(
        n_nodes,
        **_make_topo_args(kwargs, memory_capacity_factor=2),
    )
    return _build_network_finish(topo, kwargs)


def build_tree_network(
    height=2,
    **kwargs: Unpack[BuildNetworkArgs],
) -> tuple[QuantumNetwork, Simulator]:
    """
    If height==2, build the following topology:

        n4           n6
        |            |
        +n2---n1---n3+
        |            |
        n5           n7

    If height==3, build the following topology:

             n8           n12
             |            |
        n9---n4           n6---n13
             |            |
             +n2---n1---n3+
             |            |
        n10--n5           n7---n14
             |            |
             n11          n15
    """
    if height == 2:
        nnodes = 7
    elif height == 3:
        nnodes = 15
    else:
        raise ValueError("unsupported height")
    topo = TreeTopology(
        nodes_number=nnodes,
        children_number=2,
        **_make_topo_args(kwargs, memory_capacity_factor=3),
    )
    return _build_network_finish(topo, kwargs)


def build_rect_network(
    **kwargs: Unpack[BuildNetworkArgs],
) -> tuple[QuantumNetwork, Simulator]:
    """
    Build the following topology:

        n1---n2
        |     |
        n3---n4

    The network uses Yen routing algorithm with 2 paths.
    """
    topo = GridTopology(
        (2, 2),
        **_make_topo_args(kwargs, memory_capacity_factor=2),
    )
    return _build_network_finish(topo, kwargs, route=YenRouteAlgorithm(k_paths=2))


def collect_cpacket_counts(monkeypatch: pytest.MonkeyPatch, *, inc_controller=False) -> Mapping[str, int]:
    """
    Gather classic packet counts between node pairs.

    Returns: Mapping from node name(s) to number of packets, with these entries:

    * "src-dst": Packets sent from src to dst.
    * "src-*": Packets sent from src to any destination.
    * "*-dst": Packets sent to dst from any source.
    """
    orig_send_cpacket = Node.send_cpacket
    d = defaultdict[str, int](lambda: 0)

    def send_cpacket(self: Node, next_hop: Node, pkt: ClassicPacket):
        if self == pkt.src and (inc_controller or Controller not in (type(pkt.src), type(pkt.dest))):
            d[f"{pkt.src.name}-{pkt.dest.name}"] += 1
            d[f"{pkt.src.name}-*"] += 1
            d[f"*-{pkt.dest.name}"] += 1
        orig_send_cpacket(self, next_hop, pkt)

    monkeypatch.setattr(Node, "send_cpacket", send_cpacket)
    return d


def print_fw_counters(net: QuantumNetwork):
    for node in net.nodes:
        fw = node.get_app(Forwarder)
        print(node.name, fw.cnt)


def check_fw_counters(net: QuantumNetwork, **kwargs: Iterable[int | Iterable[int]]) -> None:
    __tracebackhide__ = True
    errors: list[str] = []
    cnts = [(node.name, node.get_app(Forwarder).cnt) for node in net.nodes]
    for key, expected_vec in kwargs.items():
        for expected, (node, cnt) in zip(expected_vec, cnts):
            actual = getattr(cnt, key)
            if actual != expected and (not isinstance(expected, Iterable) or actual not in expected):
                errors.append(f"{node}.{key} = {actual} != {expected}")
    assert not errors, "\n".join(errors)


def install_path(
    net: QuantumNetwork,
    rp: RoutingPath,
    *,
    t_install: float | None = 0.0,
    t_uninstall: float | None = None,
) -> RoutingPath:
    """
    Install and/or uninstall a routing path at specific times.
    """
    simulator = net.simulator
    ctrl = net.get_controller().get_app(ProactiveRoutingController)

    if t_install is not None:
        simulator.add_event(func_to_event(simulator.time(sec=t_install), ctrl.install_path, rp))

    if t_uninstall is not None:
        simulator.add_event(func_to_event(simulator.time(sec=t_uninstall), ctrl.uninstall_path, rp))

    return rp


_provide_entanglements_autoid = AutoIncrementIdentifier("Tpe_")


def provide_entanglements(
    *etgs: tuple[float, Forwarder, Forwarder],
    fidelity=0.99,
):
    """
    Provide elementary entanglement(s) to the forwarders.

    Args:
        etgs: entanglement creation time, forwarder on left side, forwarder on right side.
        fidelity: initial fidelity.
    """
    for t, src, dst in etgs:
        if t < 0:
            continue
        simulator = src.simulator
        ch = src.node.get_qchannel(dst.node)

        ch.link_arch.set(
            ch=ch,
            eta_s=1,
            eta_d=1,
            reset_time=0,
            tau_0=0,
            epr_type=src.network.epr_type,
            init_fidelity=1.0,
        )
        _, d_notify_a, d_notify_b = ch.link_arch.delays(1)

        ll_key = _provide_entanglements_autoid()
        t_creation = simulator.time(sec=t)
        epr = src.network.epr_type(
            decohere_time=t_creation + min(src.memory.t_decohere, dst.memory.t_decohere),
            fidelity_time=t_creation,
            src=src.node,
            dst=dst.node,
            mem_keys=(ll_key, ll_key),
            store_decays=(src.memory.time_decay, dst.memory.time_decay),
        )
        epr.fidelity = fidelity

        for node, neighbor, d_notify in (src, dst, d_notify_a), (dst, src, d_notify_b):
            q, _ = next(node.memory.find(lambda _, v: v is None, qchannel=ch), (None, None))
            assert q is not None, f"insufficient qubits assigned to {ch}"
            q._state = QubitState.ENTANGLED0
            q.key = ll_key
            node.memory.write(q.addr, epr)
            simulator.add_event(QubitEntangledEvent(node.node, neighbor.node, q, t=t_creation + d_notify))


def check_memory_released(net: QuantumNetwork) -> None:
    """
    Verify that all MemoryQubits on every node are in RAW or RELEASED state.
    """
    errors: list[str] = []
    for node in net.nodes:
        for qubit, data in node.memory.find(lambda *_: True):
            if qubit.state not in (QubitState.RAW, QubitState.RELEASE):
                errors.append(f"{node.name}:{qubit.addr} unexpected state {qubit.state}: {data}")
    if len(errors) > 0:
        raise AssertionError(errors)
