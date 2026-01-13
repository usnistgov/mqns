from typing import TypedDict, Unpack

from mqns.entity.cchannel import ClassicChannelInitKwargs
from mqns.entity.memory import QubitState
from mqns.entity.node import Application, Controller
from mqns.entity.qchannel import LinkArchAlways, LinkArchDimBk, QuantumChannelInitKwargs
from mqns.models.epr import Entanglement, WernerStateEntanglement
from mqns.network.network import QuantumNetwork, TimingMode, TimingModeAsync
from mqns.network.proactive import (
    LinkLayer,
    MuxScheme,
    MuxSchemeBufferSpace,
    ProactiveForwarder,
    ProactiveRoutingController,
    RoutingPath,
)
from mqns.network.protocol.event import QubitEntangledEvent
from mqns.network.route import RouteAlgorithm, YenRouteAlgorithm
from mqns.network.topology import ClassicTopology, GridTopology, LinearTopology, Topology, TopologyInitKwargs, TreeTopology
from mqns.simulator import Simulator, func_to_event
from mqns.utils import log

dflt_qchannel_args = QuantumChannelInitKwargs(
    length=100,  # delay is 0.0005 seconds
    link_arch=LinkArchAlways(LinkArchDimBk()),  # etg creation in 0.001 seconds and arrival in 0.002 seconds
)

dflt_cchannel_args = ClassicChannelInitKwargs(
    length=100,  # delay is 0.0005 seconds
)


class BuildNetworkArgs(TypedDict, total=False):
    t_cohere: float  # memory dephasing time, defaults to 5.0 seconds
    qchannel_capacity: int  # quantum channel capacity, defaults to 1
    qchannel_args: QuantumChannelInitKwargs
    cchannel_args: ClassicChannelInitKwargs
    ps: float  # probability of successful swap, defaults to 0.5
    mux: MuxScheme  # multiplexing scheme, defaults to buffer-space
    end_time: float  # simulation end time, defaults to 10.0 seconds
    timing: TimingMode  # network timing mode, defaults to ASYNC
    epr_type: type[Entanglement]  # entanglement type, defaults to werner state
    has_link_layer: bool  # whether to include full LinkLayer application, defaults to False
    init_fidelity: float  # initial fidelity, defaults to 0.99


def _make_topo_args(d: BuildNetworkArgs, *, memory_capacity_factor: int) -> TopologyInitKwargs:
    qchannel_capacity = d.get("qchannel_capacity", 1)
    nodes_apps: list[Application] = []
    if d.get("has_link_layer", False):
        nodes_apps.append(LinkLayer(init_fidelity=d.get("init_fidelity", 0.99)))
    nodes_apps.append(ProactiveForwarder(ps=d.get("ps", 0.5), mux=d.get("mux", MuxSchemeBufferSpace())))

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
    qchannel_capacity = d.get("qchannel_capacity", 1)

    topo.controller = Controller("ctrl", apps=[ProactiveRoutingController()])

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


def print_fw_counters(net: QuantumNetwork):
    for node in net.nodes:
        fw = node.get_app(ProactiveForwarder)
        print(node.name, fw.cnt)


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


def provide_entanglements(
    *etgs: tuple[float, ProactiveForwarder, ProactiveForwarder],
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
        _, d_notify_a, d_notify_b = ch.link_arch.delays(
            1,
            reset_time=0.0,
            tau_l=ch.delay_model.calculate(),
            tau_0=0.0,
        )
        t_creation = simulator.time(sec=t)
        epr = WernerStateEntanglement(
            fidelity=fidelity,
            creation_time=t_creation,
            decoherence_time=t_creation + min(src.memory.decoherence_delay, dst.memory.decoherence_delay),
            src=src.node,
            dst=dst.node,
            mem_decohere_rate=(src.memory.decoherence_rate, dst.memory.decoherence_rate),
        )
        for node, neighbor, d_notify in (src, dst, d_notify_a), (dst, src, d_notify_b):
            q, _ = next(node.memory.find(lambda _, v: v is None, qchannel=ch))
            node.memory.write(q.addr, epr)
            q._state = QubitState.ENTANGLED0
            simulator.add_event(QubitEntangledEvent(node.node, neighbor.node, q, t=t_creation + d_notify))
