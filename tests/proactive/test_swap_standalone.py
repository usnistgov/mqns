"""
Test suite for ProactiveForwarder focused on swapping.
"""

import itertools

import pytest

from mqns.models.epr import WernerStateEntanglement
from mqns.network.network import TimingModeSync
from mqns.network.proactive import (
    Fib,
    MuxSchemeDynamicEpr,
    ProactiveForwarder,
    ProactiveRoutingController,
    QubitAllocationType,
    RoutingPathMulti,
    RoutingPathSingle,
)
from mqns.simulator import func_to_event

from .proactive_common import (
    build_dumbbell_network,
    build_linear_network,
    build_rect_network,
    install_path,
    provide_entanglements,
)


def test_3_disabled():
    """Test swap disabled mode."""
    net, simulator = build_linear_network(3, ps=1.0, has_link_layer=False)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    rp = RoutingPathSingle("n1", "n3", swap=[0, 0, 0])
    install_path(ctrl, rp)

    def check_fib_entries():
        for fw in (f1, f2, f3):
            fib_entry = fw.fib.get(rp.path_id)
            assert fib_entry.is_swap_disabled is True

    simulator.add_event(func_to_event(simulator.time(sec=2.0), check_fib_entries))
    provide_entanglements(
        (1.001, f1, f2),
        (1.002, f2, f3),
    )
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)

    assert f1.cnt.n_consumed == 1
    assert f2.cnt.n_consumed == 2
    assert f3.cnt.n_consumed == 1
    assert f2.cnt.n_swapped == 0


@pytest.mark.parametrize(
    ("t_ext", "expected"),
    [
        # 1. Elementary entanglements arrive during EXTERNAL phase.
        # 2. n1-n2-n3 is swapped at t=1.008400s when INTERNAL phase begins.
        # 3. n1 and n3 are informed at t=1.008900s.
        # 4. n1-n3-n4 is swapped at t=1.008900s.
        # 5. n4 is informed at t=1.009400s and consumes the end-to-end entanglement.
        # 6. n1 is informed at t=1.009900s and consumes the end-to-end entanglement.
        (0.008400, (1, 1, 1, 1)),
        # 1. Elementary entanglements arrive during EXTERNAL phase.
        # 2. n1-n2-n3 is swapped at t=1.008900s when INTERNAL phase begins.
        # 3. n1 and n3 are informed at t=1.009400s.
        # 4. n1-n3-n4 is swapped at t=1.009400s.
        # 5. n4 is informed at t=1.009900s and consumes the end-to-end entanglement.
        # 6. n1 is informed at t=1.010400s but INTERNAL phase has ended.
        (0.008900, (0, 1, 1, 1)),
        # 1. Elementary entanglements arrive during EXTERNAL phase.
        # 2. n1-n2-n3 is swapped at t=1.009400s when INTERNAL phase begins.
        # 3. n1 and n3 are informed at t=1.009900s.
        # 4. n1-n3-n4 is swapped at t=1.009900s.
        # 5. n4 is informed at t=1.010400s but INTERNAL phase has ended.
        # 6. n1 is informed at t=1.010900s but INTERNAL phase has ended.
        (0.009400, (0, 1, 1, 0)),
        # 1. Elementary entanglements arrive during EXTERNAL phase.
        # 2. n1-n2-n3 is swapped at t=1.009900s when INTERNAL phase begins.
        # 3. n1 and n3 are informed at t=1.010400s but INTERNAL phase has ended.
        (0.009900, (0, 1, 0, 0)),
    ],
)
def test_4_sync(t_ext: float, expected: tuple[int, int, int, int]):
    """Test TimingModeSync in 4-node topology."""
    timing = TimingModeSync(t_ext=t_ext, t_int=0.010000 - t_ext)
    net, simulator = build_linear_network(4, ps=1.0, has_link_layer=False, timing=timing)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathSingle("n1", "n4", swap=[2, 0, 1, 2]))
    provide_entanglements(
        (1.001, f1, f2),
        (1.001, f2, f3),
        (1.001, f3, f4),
    )
    simulator.run()

    for fw in (f1, f2, f3, f4):
        print(fw.own.name, fw.cnt)

    assert (f1.cnt.n_consumed, f2.cnt.n_swapped, f3.cnt.n_swapped, f4.cnt.n_consumed) == expected


@pytest.mark.parametrize(
    ("arrival_ms", "n_swapped_p"),
    [
        ((1, 2, 1), 1),
        ((2, 1, 2), 1),
        ((1, 2, 3), 0),
        ((3, 2, 1), 0),
    ],
)
def test_4_asap(arrival_ms: tuple[int, int, int], n_swapped_p: int):
    """Test SWAP-ASAP in 4-node topology with various entanglement arrival orders."""
    net, simulator = build_linear_network(4, ps=1.0, has_link_layer=False)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathSingle("n1", "n4", swap=[1, 0, 0, 1]))
    provide_entanglements(
        (1 + arrival_ms[0] / 1000, f1, f2),
        (1 + arrival_ms[1] / 1000, f2, f3),
        (1 + arrival_ms[2] / 1000, f3, f4),
    )
    simulator.run()

    for fw in (f1, f2, f3, f4):
        print(fw.own.name, fw.cnt)

    assert f1.cnt.n_consumed == 1 == f4.cnt.n_consumed
    assert f2.cnt.n_swapped_s == 1 == f3.cnt.n_swapped_s
    assert f2.cnt.n_swapped_p == n_swapped_p == f3.cnt.n_swapped_p


@pytest.mark.parametrize(
    ("ps3", "arrival_ms", "n_swapped_s", "n_swapped_p", "n_consumed"),
    [
        # 1. n2-n3-n4 swap succeeds.
        # 2. n2 and n4 are informed.
        # 3. n1-n2-n4 and n2-n4-n5 swaps succeed sequentially.
        # 4. n1-n2-n4 and n2-n4-n5 swaps succeed in parallel.
        (1.0, (2, 1, 1, 2), (1, 1, 1), (1, 0, 1), 1),
        # 1. n2-n3-n4 swap fails.
        # 2. n2 and n4 are informed.
        # 3. There's nothing to swap with n1-n2 and n4-n5.
        (0.0, (2, 1, 1, 2), (0, 0, 0), (0, 0, 0), 0),
        # 1. n1-n2-n3 and n2-n3-n4 and n3-n4-n5 swaps succeed in parallel.
        (1.0, (1, 2, 2, 1), (1, 1, 1), (2, 2, 2), 1),
        # 1. n1-n2-n3 and n2-n3-n4 and n3-n4-n5 swaps are attempted in parallel.
        #    n1-n2-n3 and n3-n4-n5 swaps succeed, but n2-n3-n4 swap fails.
        (0.0, (1, 2, 2, 1), (1, 0, 1), (0, 0, 0), 0),
    ],
)
def test_5_asap(
    ps3: float,
    arrival_ms: tuple[int, int, int, int],
    n_swapped_s: tuple[int, int, int],
    n_swapped_p: tuple[int, int, int],
    n_consumed: int,
):
    """Test SWAP-ASAP in 5-node topology with various entanglement arrival orders."""
    net, simulator = build_linear_network(5, ps=1.0, has_link_layer=False)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)
    f3.ps = ps3

    install_path(ctrl, RoutingPathSingle("n1", "n5", swap=[1, 0, 0, 0, 1]))
    provide_entanglements(
        (1 + arrival_ms[0] / 1000, f1, f2),
        (1 + arrival_ms[1] / 1000, f2, f3),
        (1 + arrival_ms[2] / 1000, f3, f4),
        (1 + arrival_ms[3] / 1000, f4, f5),
    )
    simulator.run()

    for fw in (f1, f2, f3, f4, f5):
        print(fw.own.name, fw.cnt)

    assert f1.cnt.n_consumed == n_consumed == f5.cnt.n_consumed
    assert (f2.cnt.n_swapped_s, f3.cnt.n_swapped_s, f4.cnt.n_swapped_s) == n_swapped_s
    assert (f2.cnt.n_swapped_p, f3.cnt.n_swapped_p, f4.cnt.n_swapped_p) == n_swapped_p


@pytest.mark.parametrize(
    ("swap", "arrival_ms"),
    itertools.product(
        (
            [3, 0, 1, 2, 3],  # l2r
            [3, 2, 1, 0, 3],  # r2l
            [3, 0, 1, 0, 3],  # baln
        ),
        itertools.permutations(range(4), 4),
    ),
)
def test_5_sequential(swap: list[int], arrival_ms: tuple[int, int, int, int]):
    """Test sequential swap orders with various entanglement arrival orders."""
    net, simulator = build_linear_network(5, ps=1.0, has_link_layer=False)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathSingle("n1", "n5", swap=swap))
    provide_entanglements(
        (1 + arrival_ms[0] / 1000, f1, f2),
        (1 + arrival_ms[1] / 1000, f2, f3),
        (1 + arrival_ms[2] / 1000, f3, f4),
        (1 + arrival_ms[3] / 1000, f4, f5),
    )
    simulator.run()

    for fw in (f1, f2, f3, f4, f5):
        print(fw.own.name, fw.cnt)

    assert f1.cnt.n_consumed == 1 == f5.cnt.n_consumed
    assert (f2.cnt.n_swapped_s, f3.cnt.n_swapped_s, f4.cnt.n_swapped_s) == (1, 1, 1)
    assert (f2.cnt.n_swapped_p, f3.cnt.n_swapped_p, f4.cnt.n_swapped_p) == (0, 0, 0)


@pytest.mark.parametrize(
    ("has_etg", "n_swapped", "n_consumed"),
    [
        ((0, 1, 0, 1), (0, 0), 0),
        ((1, 0, 1, 0), (0, 0), 0),
        ((1, 1, 0, 0), (1, 0), 1),
        ((0, 0, 1, 1), (0, 1), 1),
        ((1, 1, 1, 1), (1, 1), 2),
    ],
)
def test_rect_multipath(has_etg: tuple[int, int, int, int], n_swapped: tuple[int, int], n_consumed: int):
    """Test swapping in rectangular network with a multi-path request."""
    net, simulator = build_rect_network(ps=1.0, has_link_layer=False)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    rp = RoutingPathMulti("n1", "n4", swap=[1, 0, 1])
    install_path(ctrl, rp)

    def check_fib_entries():
        routes = {"-".join(f1.fib.get(path_id).route) for path_id in (rp.path_id, rp.path_id + 1)}
        assert routes == {"n1-n2-n4", "n1-n3-n4"}

    simulator.add_event(func_to_event(simulator.time(sec=2.0), check_fib_entries))
    provide_entanglements(
        (1.001 if has_etg[0] else -1, f1, f2),
        (1.002 if has_etg[1] else -1, f2, f4),
        (1.001 if has_etg[2] else -1, f1, f3),
        (1.002 if has_etg[3] else -1, f3, f4),
    )
    simulator.run()

    for fw in (f1, f2, f3, f4):
        print(fw.own.name, fw.cnt)

    assert f1.cnt.n_consumed == n_consumed == f4.cnt.n_consumed
    assert (f2.cnt.n_swapped, f3.cnt.n_swapped) == n_swapped


@pytest.mark.parametrize(
    ("t_edge_etg", "selected_path", "n_consumed"),
    [
        # Both n2-n1 and n1-n3 channels select the same path,
        # so that end-to-end entanglements are created on the selected path,
        # regardless of whether edge entanglements arrive before or after center entanglements.
        (1.001, (0, 0), (1, 0)),
        (1.001, (1, 1), (0, 1)),
        (1.007, (0, 0), (1, 0)),
        (1.007, (1, 1), (0, 1)),
        # The n2-n1 and n1-n3 channels select different paths,
        # so that end-to-end entanglements are not created,
        # regardless of whether edge entanglements arrive before or after center entanglements.
        (1.001, (0, 1), (0, 0)),
        (1.001, (1, 0), (0, 0)),
        (1.007, (0, 1), (0, 0)),
        (1.007, (1, 0), (0, 0)),
    ],
)
def test_dumbbell_dynepr(t_edge_etg: float, selected_path: tuple[int, int], n_consumed: tuple[int, int]):
    """Test MuxSchemeDynamicEpr in dumbbell network."""

    def select_path(epr: WernerStateEntanglement, fib: Fib, path_ids: list[int]) -> int:
        _ = fib
        if len(path_ids) != 2:
            chosen = path_ids[0]
        elif epr.src is f2.own:  # n2-n1
            chosen = (rp0.path_id, rp1.path_id)[selected_path[0]]
        else:  # n1-n3
            assert epr.src is f1.own
            chosen = (rp0.path_id, rp1.path_id)[selected_path[1]]
        return chosen

    net, simulator = build_dumbbell_network(
        ps=1.0,
        has_link_layer=False,
        mux=MuxSchemeDynamicEpr(select_path=select_path),
    )
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f6 = net.get_node("n6").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)
    f7 = net.get_node("n7").get_app(ProactiveForwarder)

    # n4-n2-n1-n3-n6
    rp0 = RoutingPathSingle("n4", "n6", qubit_allocation=QubitAllocationType.DISABLED, swap="asap")
    install_path(ctrl, rp0)
    # n5-n2-n1-n3-n7
    rp1 = RoutingPathSingle("n5", "n7", qubit_allocation=QubitAllocationType.DISABLED, swap="asap")
    install_path(ctrl, rp1)

    provide_entanglements(
        (t_edge_etg, f4, f2),
        (t_edge_etg, f5, f2),
        (t_edge_etg, f3, f6),
        (t_edge_etg, f3, f7),
        (1.003, f2, f1),
        (1.005, f1, f3),
    )
    simulator.run()

    for fw in (f1, f2, f3, f4, f5, f6, f7):
        print(fw.own.name, fw.cnt)

    assert f4.cnt.n_consumed == n_consumed[0] == f6.cnt.n_consumed
    assert f5.cnt.n_consumed == n_consumed[1] == f7.cnt.n_consumed
