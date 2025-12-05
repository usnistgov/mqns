import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from mqns.entity.cchannel import ClassicChannelInitKwargs
from mqns.entity.node import Controller
from mqns.entity.qchannel import LinkArchAlways, LinkArchDimBk, QuantumChannelInitKwargs
from mqns.network.network import QuantumNetwork, TimingMode, TimingModeAsync, TimingModeSync
from mqns.network.proactive import (
    CutoffSchemeWaitTime,
    LinkLayer,
    MuxScheme,
    MuxSchemeBufferSpace,
    MuxSchemeDynamicEpr,
    ProactiveForwarder,
    ProactiveRoutingController,
    QubitAllocationType,
    RoutingPath,
    RoutingPathMulti,
    RoutingPathSingle,
    RoutingPathStatic,
)
from mqns.network.proactive.message import validate_path_instructions
from mqns.network.route import RouteImpl, YenRouteAlgorithm
from mqns.network.topology import ClassicTopology, GridTopology, LinearTopology, Topology, TreeTopology
from mqns.simulator import Simulator, func_to_event
from mqns.utils import log

init_fidelity = 0.90

dflt_qchannel_args = QuantumChannelInitKwargs(
    length=100,  # delay is 0.0005 seconds
    link_arch=LinkArchAlways(LinkArchDimBk()),  # entanglement in 0.002 seconds
)

dflt_cchannel_args = ClassicChannelInitKwargs(
    length=100,  # delay is 0.0005 seconds
)


def build_network_finish(
    topo: Topology,
    qchannel_capacity: int,
    end_time: float,
    timing: TimingMode,
    *,
    route: RouteImpl | None = None,
):
    topo.controller = Controller("ctrl", apps=[ProactiveRoutingController()])

    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow, route=route, timing=timing)
    for qchannel in net.qchannels:
        qchannel.assign_memory_qubits(capacity=qchannel_capacity)
    topo.connect_controller(net.nodes)

    simulator = Simulator(0.0, end_time)
    log.install(simulator)
    net.install(simulator)

    return net, simulator


def build_linear_network(
    n_nodes: int,
    *,
    qchannel_capacity=1,
    qchannel_args=dflt_qchannel_args,
    cchannel_args=dflt_cchannel_args,
    mux: MuxScheme = MuxSchemeBufferSpace(),
    end_time=10.0,
    timing: TimingMode = TimingModeAsync(),
) -> tuple[QuantumNetwork, Simulator]:
    topo = LinearTopology(
        n_nodes,
        nodes_apps=[
            LinkLayer(init_fidelity=init_fidelity),
            ProactiveForwarder(ps=0.5, mux=mux),
        ],
        qchannel_args=qchannel_args,
        cchannel_args=cchannel_args,
        memory_args={"decoherence_rate": 1 / 5.0, "capacity": 2 * qchannel_capacity},
    )
    return build_network_finish(topo, qchannel_capacity, end_time, timing)


def build_dumbbell_network(
    *,
    qchannel_capacity=1,
    qchannel_args=dflt_qchannel_args,
    cchannel_args=dflt_cchannel_args,
    mux: MuxScheme = MuxSchemeBufferSpace(),
    end_time=10.0,
    timing: TimingMode = TimingModeAsync(),
) -> tuple[QuantumNetwork, Simulator]:
    """
    Build the following topology:

        n4           n6
        |            |
        +n2---n1---n3+
        |            |
        n5           n7
    """
    topo = TreeTopology(
        nodes_number=7,
        children_number=2,
        nodes_apps=[
            LinkLayer(init_fidelity=init_fidelity),
            ProactiveForwarder(ps=0.5, mux=mux),
        ],
        qchannel_args=qchannel_args,
        cchannel_args=cchannel_args,
        memory_args={"decoherence_rate": 1 / 5.0, "capacity": 3 * qchannel_capacity},
    )
    return build_network_finish(topo, qchannel_capacity, end_time, timing)


def build_rect_network(
    *,
    qchannel_capacity=1,
    qchannel_args=dflt_qchannel_args,
    cchannel_args=dflt_cchannel_args,
    mux: MuxScheme = MuxSchemeBufferSpace(),
    end_time=10.0,
    timing: TimingMode = TimingModeAsync(),
) -> tuple[QuantumNetwork, Simulator]:
    """
    Build the following topology:

        n1---n2
        |     |
        n3---n4
    """
    topo = GridTopology(
        (2, 2),
        nodes_apps=[
            LinkLayer(init_fidelity=init_fidelity),
            ProactiveForwarder(ps=0.5, mux=mux),
        ],
        qchannel_args=qchannel_args,
        cchannel_args=cchannel_args,
        memory_args={"decoherence_rate": 1 / 5.0, "capacity": 2 * qchannel_capacity},
    )
    return build_network_finish(topo, qchannel_capacity, end_time, timing, route=YenRouteAlgorithm(k_paths=2))


def install_path(
    ctrl: ProactiveRoutingController,
    rp: RoutingPath,
    *,
    t_install: float | None = 0.0,
    t_uninstall: float | None = None,
):
    """
    Install and/or uninstall a routing path at specific times.
    """
    simulator = ctrl.own.simulator

    if t_install is not None:
        simulator.add_event(func_to_event(simulator.time(sec=t_install), ctrl.install_path, rp))

    if t_uninstall is not None:
        simulator.add_event(func_to_event(simulator.time(sec=t_uninstall), ctrl.uninstall_path, rp))


class CheckUnchanged:
    """
    Check one or more counters are unchanged during a period of time.

    Example:
    ```
    with CheckUnchanged(...):
        simulator.run()
    ```
    """

    def __init__(
        self,
        simulator: Simulator,
        t0: float,
        t1: float,
        getter: Callable[[], Any],
        *,
        abs=1e-6,
    ):
        self.values: list[Any] = []
        self.abs = abs
        simulator.add_event(func_to_event(simulator.time(sec=t0), lambda: self.values.append(getter())))
        simulator.add_event(func_to_event(simulator.time(sec=t1), lambda: self.values.append(getter())))

    def __enter__(self):
        pass

    def __exit__(self, *_):
        v0, v1 = self.values
        assert v0 == pytest.approx(v1, abs=self.abs)


def check_e2e_consumed(
    fl: ProactiveForwarder,
    fr: ProactiveForwarder,
    *,
    n_swaps: int | None = None,
    n_min=0,
    n_max=math.inf,
    swap_balanced=False,
    has_purif=False,
    capacity=1,
    f_min=0.0,
    f_max=1.0,
):
    """
    Check consumption counters of an end-to-end path.

    Args:
        fl: leftmost forwarder.
        fr: rightmost forwarder.
        n_swaps: swap counter at the repeater that performs final swaps, if known.
        n_min: minimum acceptable count, ignored if `n_swaps` is specified.
        n_max: maximum acceptable count, ignored if `n_swaps` is specified.
        swap_balanced: whether the swapping order was "balanced".
        has_purif: whether there is purification.
        capacity: how many qubits were assigned to this path at either `fl` or `fr`
                  (if different, pass the lesser value).
        f_min: minimum acceptable fidelity.
        f_max: maximum acceptable fidelity.
    """
    if n_swaps is not None:
        n_min = n_swaps - capacity
        n_max = n_swaps

    # every eligible qubit is immediately consumed
    assert n_max >= fl.cnt.n_eligible == fl.cnt.n_consumed >= n_min
    assert n_max >= fr.cnt.n_eligible == fr.cnt.n_consumed >= n_min

    # fidelity should be within range
    assert f_min <= fl.cnt.consumed_avg_fidelity <= f_max
    assert f_min <= fr.cnt.consumed_avg_fidelity <= f_max

    # If the swapping order is not balanced, some SWAP_UPDATE messages may still be in-flight at end of simulation,
    # so that the consumption counters can be slightly different.
    # If purification is enabled on the path, some PURIF_RESPONSE messages may still be in-flight at end of simulation,
    # so that the consumption counter on the left side can be slightly less than it on the right side.
    # In either case, the difference should never exceed how many qubits were assigned to this path.

    if swap_balanced:
        if has_purif:
            assert fr.cnt.n_consumed >= fl.cnt.n_consumed >= fr.cnt.n_consumed - capacity
        else:
            assert fl.cnt.n_consumed == fr.cnt.n_consumed
            assert fl.cnt.consumed_avg_fidelity == pytest.approx(fr.cnt.consumed_avg_fidelity, abs=1e-6)
            return
    else:
        assert fl.cnt.n_consumed == pytest.approx(fr.cnt.n_consumed, abs=capacity)

    if min(fl.cnt.n_consumed, fr.cnt.n_consumed) >= 100:
        assert fl.cnt.consumed_avg_fidelity == pytest.approx(fr.cnt.consumed_avg_fidelity, abs=1e-2)


def test_path_validation():
    """Test path validation logic."""

    route3 = ["n1", "n2", "n3"]
    swap3 = [1, 0, 1]
    scut3 = [-1, 1000, -1]
    mv3 = [(1, 1)] * 2

    with pytest.raises(ValueError, match="route is empty"):
        validate_path_instructions({"req_id": 0, "route": [], "swap": [], "swap_cutoff": [], "purif": {}})

    with pytest.raises(ValueError, match="swapping order"):
        validate_path_instructions(
            {"req_id": 0, "route": ["n1", "n2", "n3", "n4", "n5"], "swap": swap3, "swap_cutoff": scut3, "purif": {}}
        )

    with pytest.raises(ValueError, match="swap_cutoff"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": [-1, 1000, 1000, -1], "purif": {}}
        )

    with pytest.raises(ValueError, match="multiplexing vector"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": [(1, 1)] * 3, "purif": {}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"r1-r2": 1}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"n1-n2-n3": 1}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"n2-n2": 1}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"n3-n1": 1}}
        )


def test_no_swap():
    """Test isolated links mode where swapping is disabled."""
    net, simulator = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3"], swap=[0, 0, 0]))
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)
        assert 0.88 <= fw.cnt.consumed_avg_fidelity <= 0.90

    assert f1.cnt.n_entg == f1.cnt.n_eligible == f1.cnt.n_consumed == pytest.approx(5000, abs=5)
    assert f2.cnt.n_entg == f2.cnt.n_eligible == f2.cnt.n_consumed == pytest.approx(10000, abs=5)
    assert f3.cnt.n_entg == f3.cnt.n_eligible == f3.cnt.n_consumed == pytest.approx(5000, abs=5)
    assert f1.cnt.n_swapped == f2.cnt.n_swapped == f3.cnt.n_swapped == 0
    assert f1.cnt.n_consumed + f3.cnt.n_consumed == f2.cnt.n_consumed


def test_swap_1():
    """Test basic 1-repeater swapping."""
    net, simulator = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1]))
    f3.cnt.enable_collect_all()
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)
    assert f1.cnt.consumed_fidelity_values is None
    print(np.histogram(f3.cnt.consumed_fidelity_values or [], bins=4))

    # entanglements at n2 are immediately eligible because there's no purification
    assert f1.cnt.n_entg + f3.cnt.n_entg == f2.cnt.n_entg == f2.cnt.n_eligible == pytest.approx(8000, abs=5)
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert 0 < f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.7
    # no swapping is expected at n1 and n3
    assert f1.cnt.n_swapped == f3.cnt.n_swapped == 0
    # successful swap at n2 should make the qubit eligible at n1 and n3, but allow small loss at end of simulation
    assert f2.cnt.n_swapped >= f1.cnt.n_eligible >= f2.cnt.n_swapped - 2
    # consumptions are expected at n1 and n3;
    # consumed entanglements should have expected fidelity worse than initial fidelity
    check_e2e_consumed(f1, f3, n_swaps=f2.cnt.n_swapped, swap_balanced=True, f_min=0.78, f_max=0.84)
    # no consumption is expected at n2
    assert f2.cnt.n_consumed == 0


def test_swap_2():
    """Test 2-repeater parallel swapping."""
    net, simulator = build_linear_network(4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3", "n4"], swap=[1, 0, 0, 1]))
    simulator.run()

    for fw in (f1, f2, f3, f4):
        print(fw.own.name, fw.cnt)

    # entanglements at n2 and n3 are immediately eligible because there's no purification
    assert f2.cnt.n_entg == f2.cnt.n_eligible > 6000
    assert f3.cnt.n_entg == f3.cnt.n_eligible > 6000
    # only eligible qubits may be swapped at n2 and n3, with 50% success rate
    assert 0 < f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.7
    assert 0 < f3.cnt.n_swapped <= f3.cnt.n_eligible * 0.7
    # some swaps were completed in parallel
    assert f2.cnt.n_swapped_p > 0
    assert f3.cnt.n_swapped_p > 0
    # successful swap at both n2 and n3 can make the qubit eligible at n1 and n4,
    # but there would be losses because parallel swaps require coincidence;
    # consumed entanglements should have expected fidelity worse than initial fidelity
    check_e2e_consumed(f1, f4, n_min=600, n_max=min(f2.cnt.n_swapped, f3.cnt.n_swapped) - 1, f_min=0.71, f_max=0.77)


def run_swap_3(timing: TimingMode):
    net, simulator = build_linear_network(5, timing=timing, end_time=20.0)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3", "n4", "n5"], swap=[2, 0, 1, 0, 2]), t_uninstall=9.999999)
    with CheckUnchanged(simulator, 10.0, 20.0, lambda: (f1.cnt.n_consumed, f5.cnt.n_consumed, f3.cnt.n_swapped)):
        simulator.run()

    for fw in (f1, f2, f3, f4, f5):
        print(fw.own.name, fw.cnt)
        assert fw.cnt.n_swapped_p == 0

    return f1, f2, f3, f4, f5


def test_swap_3_async():
    """Test 3-repeater sequential swapping, using ASYNC timing mode."""
    f1, f2, f3, f4, f5 = run_swap_3(TimingModeAsync())

    assert 2400 < f1.cnt.n_entg < 2800
    assert 4800 < f2.cnt.n_entg < 5600
    assert 4800 < f3.cnt.n_entg < 5600
    assert 4800 < f4.cnt.n_entg < 5600
    assert 2400 < f5.cnt.n_entg < 2800
    assert 1000 < f2.cnt.n_swapped < 1600
    assert 1000 < f4.cnt.n_swapped < 1600
    assert 500 < f3.cnt.n_swapped < 800
    check_e2e_consumed(f1, f5, n_swaps=f3.cnt.n_swapped, swap_balanced=True)


def test_swap_3_sync():
    """Test 3-repeater sequential swapping, using SYNC timing mode."""
    f1, f2, f3, f4, f5 = run_swap_3(
        TimingModeSync(t_ext=0.008, t_int=0.002),  # 0.01 seconds per time slot, 1000 total time slots
    )

    # first time slot is spent in processing routing commands; entanglements begin at the second time slot
    assert f1.cnt.n_entg == f5.cnt.n_entg == 999
    assert f2.cnt.n_entg == f3.cnt.n_entg == f4.cnt.n_entg == 1998
    assert 440 < f2.cnt.n_swapped < 560
    assert 440 < f4.cnt.n_swapped < 560
    assert 100 < f3.cnt.n_swapped < 160
    # final INTERNAL phase has sufficient time for consumption, can't lose any at the end
    check_e2e_consumed(f1, f5, n_swaps=f3.cnt.n_swapped, swap_balanced=True)


def test_swap_mr():
    """Test swapping over multiple requests."""
    net, simulator = build_dumbbell_network(qchannel_capacity=8, mux=MuxSchemeDynamicEpr(), end_time=90.0)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f6 = net.get_node("n6").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)
    f7 = net.get_node("n7").get_app(ProactiveForwarder)

    # n4-n2-n1-n3-n6
    install_path(ctrl, RoutingPathSingle("n4", "n6", qubit_allocation=QubitAllocationType.DISABLED, swap="l2r"), t_install=10)
    # n5-n2-n1-n3-n7
    install_path(ctrl, RoutingPathSingle("n5", "n7", qubit_allocation=QubitAllocationType.DISABLED, swap="l2r"), t_uninstall=80)

    with (
        CheckUnchanged(simulator, 0, 9, lambda: (f4.cnt.n_entg, f4.cnt.n_consumed)),
        CheckUnchanged(simulator, 81, 90, lambda: (f5.cnt.n_entg, f7.cnt.n_consumed)),
    ):
        simulator.run()

    for fw in (f4, f6, f5, f7):
        print(fw.own.name, fw.cnt)

    # some end-to-end entanglements should be consumed at n4 and n6
    check_e2e_consumed(f4, f6, n_min=1, capacity=8)
    # some end-to-end entanglements should be consumed at n5 and n7
    check_e2e_consumed(f5, f7, n_min=1, capacity=8)


def test_swap_mp():
    """Test swapping over multiple paths."""
    net, simulator = build_rect_network(qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    # n1-n2-n4 and n1-n3-n4
    install_path(ctrl, RoutingPathMulti("n1", "n4", swap="swap_1"))
    simulator.run()

    for fw in (f1, f2, f3, f4):
        print(fw.own.name, fw.cnt)

    # both paths are used
    assert f2.cnt.n_swapped > 4000
    assert f3.cnt.n_swapped > 4000
    # swapped EPRs are consumed, capacity=8 is twice of qchannel_capacity because there are two paths
    check_e2e_consumed(f1, f4, n_swaps=f2.cnt.n_swapped + f3.cnt.n_swapped, swap_balanced=True, capacity=8)


def test_cutoff_waittime():
    """Test 5-repeater swapping with wait-time cutoff."""
    qchannel_args = deepcopy(dflt_qchannel_args)
    qchannel_args["link_arch"] = LinkArchDimBk()
    net, simulator = build_linear_network(7, qchannel_capacity=1, qchannel_args=qchannel_args, end_time=300)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)
    f6 = net.get_node("n6").get_app(ProactiveForwarder)
    f7 = net.get_node("n7").get_app(ProactiveForwarder)

    for fw in (f2, f3, f4, f5, f6):
        CutoffSchemeWaitTime.of(fw).cnt.enable_collect_all()

    install_path(ctrl, RoutingPathSingle("n1", "n7", swap=[3, 0, 1, 0, 2, 0, 3], swap_cutoff=[0.5] * 7))
    simulator.run()

    for fw in (f1, f2, f3, f4, f5, f6, f7):
        print(fw.own.name, fw.cnt)

    assert f1.cnt.n_cutoff[0] == 0
    assert f7.cnt.n_cutoff[0] == 0
    assert f2.cnt.n_cutoff[1] == 0
    assert f4.cnt.n_cutoff[1] == 0
    assert f6.cnt.n_cutoff[1] == 0
    assert f2.cnt.n_cutoff[0] + f4.cnt.n_cutoff[0] + f6.cnt.n_cutoff[0] > 0
    assert f1.cnt.n_cutoff[1] + f3.cnt.n_cutoff[1] + f5.cnt.n_cutoff[1] + f7.cnt.n_cutoff[1] > 0

    for fw in (f2, f3, f4, f5, f6):
        cutoff = CutoffSchemeWaitTime.of(fw)
        print(np.histogram(cutoff.cnt.wait_values or [], bins=4))
        assert fw.cnt.n_eligible / 2 >= len(cutoff.cnt.wait_values or []) >= fw.cnt.n_swapped


def test_purif_link1r():
    """Test 1-round purification on each link."""
    net, simulator = build_linear_network(3, qchannel_capacity=2)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], purif={"n1-n2": 1, "n2-n3": 1}))
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)

        # some purifications should fail
        assert fw.cnt.n_purif[0] < fw.cnt.n_entg * 0.8

    # entanglements at n2 are eligible after 1-round purification
    f1f3_n_purif0 = f1.cnt.n_purif[0] + f3.cnt.n_purif[0]
    assert f1f3_n_purif0 >= f2.cnt.n_purif[0] == f2.cnt.n_eligible >= f1f3_n_purif0 - 2
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert f2.cnt.n_eligible * 0.7 >= f2.cnt.n_swapped >= 1000
    # successful swap at n2 should make the qubit eligible at n1 and n3;
    # eligible qubits at n1 and n3 are immediately consumed;
    # consumed entanglements should have expected fidelity better than swap_1
    check_e2e_consumed(f1, f3, n_swaps=f2.cnt.n_swapped, swap_balanced=True, has_purif=True, capacity=2, f_min=0.82, f_max=0.88)


def test_purif_link2r():
    """Test 2-round purification on each link."""
    net, simulator = build_linear_network(3, qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], purif={"n1-n2": 2, "n2-n3": 2}))
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)

        # some purifications should fail
        assert fw.cnt.n_purif[0] < fw.cnt.n_entg * 0.8
        assert fw.cnt.n_purif[1] < fw.cnt.n_purif[0] * 0.8

    # entanglements at n2 are eligible after 2-round purification
    f1f3_n_purif0 = f1.cnt.n_purif[0] + f3.cnt.n_purif[0]
    assert f1f3_n_purif0 >= f2.cnt.n_purif[0] >= f1f3_n_purif0 - 4
    f1f3_n_purif1 = f1.cnt.n_purif[1] + f3.cnt.n_purif[1]
    assert f1f3_n_purif1 >= f2.cnt.n_purif[1] == f2.cnt.n_eligible >= f1f3_n_purif1 - 4
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert 0 < f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.7
    assert f2.cnt.n_swapped > 600
    # successful swap at n2 should make the qubit eligible at n1 and n3;
    # eligible qubits at n1 and n3 are immediately consumed;
    # consumed entanglements should have expected fidelity better than purif_link1r
    check_e2e_consumed(f1, f3, n_swaps=f2.cnt.n_swapped, swap_balanced=True, has_purif=True, capacity=4, f_min=0.86, f_max=0.92)


def test_purif_ee2r():
    """Test 2-round purification between two end nodes."""
    net, simulator = build_linear_network(3, qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], purif={"n1-n3": 2}))
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)

    # successful swap at n2 enables first round of n1-n3 purification; some purifications should fail
    assert pytest.approx(f1.cnt.n_purif[0], abs=1) == f3.cnt.n_purif[0] < f2.cnt.n_swapped * 0.6
    assert pytest.approx(f1.cnt.n_purif[1], abs=1) == f3.cnt.n_purif[1] < f3.cnt.n_purif[0] * 0.6
    # no purification may occur in n2
    assert len(f2.cnt.n_purif) == 0
    # eligible qubits at n1 and n3 are consumed after completing 2-round purification;
    # consumed entanglements should have expected fidelity better than swap_1
    check_e2e_consumed(f1, f3, n_min=500, capacity=4, swap_balanced=True, has_purif=True, f_min=0.85, f_max=0.91)
