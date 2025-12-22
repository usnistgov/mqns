"""
Test suite for swapping in proactive forwarding, standalone without LinkLayer.
"""

import itertools

import pytest

from mqns.network.network import TimingModeSync
from mqns.network.proactive import (
    ProactiveForwarder,
    ProactiveRoutingController,
    RoutingPathSingle,
)
from mqns.simulator import func_to_event

from .proactive_common import (
    build_linear_network,
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

    PATH_ID = 27
    install_path(ctrl, RoutingPathSingle("n1", "n3", swap=[0, 0, 0], path_id=PATH_ID))

    def check_fib_entries():
        for fw in (f1, f2, f3):
            fib_entry = fw.fib.get(PATH_ID)
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
    """Test SWAP-ASAP in 5-node topology with a specific entanglement arrival order."""
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
