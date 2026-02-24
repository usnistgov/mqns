"""
Test suite for ProactiveForwarder focused on swapping.
"""

import itertools

import pytest

from mqns.models.epr import Entanglement
from mqns.network.fw import (
    Fib,
    Forwarder,
    MemoryEprTuple,
    MuxSchemeDynamicEpr,
    MuxSchemeStatistical,
    QubitAllocationType,
    RoutingPathMulti,
    RoutingPathSingle,
)
from mqns.network.network import TimingModeSync
from mqns.network.proactive import ProactiveForwarder
from mqns.simulator import func_to_event

from .proactive_common import (
    build_linear_network,
    build_rect_network,
    build_tree_network,
    install_path,
    print_fw_counters,
    provide_entanglements,
)


def test_3_disabled():
    """Test swap disabled mode."""
    net, simulator = build_linear_network(3, ps=1.0)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    rp = install_path(net, RoutingPathSingle("n1", "n3", swap=[0, 0, 0]))

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
    print_fw_counters(net)

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
    net, simulator = build_linear_network(4, ps=1.0, timing=timing)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    install_path(net, RoutingPathSingle("n1", "n4", swap=[2, 0, 1, 2]))
    provide_entanglements(
        (1.001, f1, f2),
        (1.001, f2, f3),
        (1.001, f3, f4),
    )
    simulator.run()
    print_fw_counters(net)

    assert (f1.cnt.n_consumed, f2.cnt.n_swapped, f3.cnt.n_swapped, f4.cnt.n_consumed) == expected


@pytest.mark.parametrize(
    ("etg_ms", "n_swapped_p"),
    [
        ((1, 2, 1), 1),
        ((2, 1, 2), 1),
        ((1, 2, 3), 0),
        ((3, 2, 1), 0),
    ],
)
def test_4_asap(etg_ms: tuple[int, int, int], n_swapped_p: int):
    """Test SWAP-ASAP in 4-node topology with various entanglement arrival orders."""
    net, simulator = build_linear_network(4, ps=1.0)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    install_path(net, RoutingPathSingle("n1", "n4", swap=[1, 0, 0, 1]))
    provide_entanglements(
        (1 + etg_ms[0] / 1000, f1, f2),
        (1 + etg_ms[1] / 1000, f2, f3),
        (1 + etg_ms[2] / 1000, f3, f4),
    )
    simulator.run()
    print_fw_counters(net)

    assert f1.cnt.n_consumed == 1 == f4.cnt.n_consumed
    assert f2.cnt.n_swapped_s == 1 == f3.cnt.n_swapped_s
    assert f2.cnt.n_swapped_p == n_swapped_p == f3.cnt.n_swapped_p


@pytest.mark.parametrize(
    ("ps3", "etg_ms", "n_swapped_s", "n_swapped_p", "n_consumed"),
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
    etg_ms: tuple[int, int, int, int],
    n_swapped_s: tuple[int, int, int],
    n_swapped_p: tuple[int, int, int],
    n_consumed: int,
):
    """Test SWAP-ASAP in 5-node topology with various entanglement arrival orders."""
    net, simulator = build_linear_network(5, ps=1.0)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)
    f3.ps = ps3

    install_path(net, RoutingPathSingle("n1", "n5", swap=[1, 0, 0, 0, 1]))
    provide_entanglements(
        (1 + etg_ms[0] / 1000, f1, f2),
        (1 + etg_ms[1] / 1000, f2, f3),
        (1 + etg_ms[2] / 1000, f3, f4),
        (1 + etg_ms[3] / 1000, f4, f5),
    )
    simulator.run()
    print_fw_counters(net)

    assert f1.cnt.n_consumed == n_consumed == f5.cnt.n_consumed
    assert (f2.cnt.n_swapped_s, f3.cnt.n_swapped_s, f4.cnt.n_swapped_s) == n_swapped_s
    assert (f2.cnt.n_swapped_p, f3.cnt.n_swapped_p, f4.cnt.n_swapped_p) == n_swapped_p


@pytest.mark.parametrize(
    ("swap", "etg_ms"),
    itertools.product(
        (
            [3, 0, 1, 2, 3],  # l2r
            [3, 2, 1, 0, 3],  # r2l
            [3, 0, 1, 0, 3],  # baln
        ),
        itertools.permutations(range(4), 4),
    ),
)
def test_5_sequential(swap: list[int], etg_ms: tuple[int, int, int, int]):
    """Test sequential swap orders with various entanglement arrival orders."""
    net, simulator = build_linear_network(5, ps=1.0)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)

    install_path(net, RoutingPathSingle("n1", "n5", swap=swap))
    provide_entanglements(
        (1 + etg_ms[0] / 1000, f1, f2),
        (1 + etg_ms[1] / 1000, f2, f3),
        (1 + etg_ms[2] / 1000, f3, f4),
        (1 + etg_ms[3] / 1000, f4, f5),
    )
    simulator.run()
    print_fw_counters(net)

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
    """Test swapping in rectangular topology with a multi-path request."""
    net, simulator = build_rect_network(ps=1.0)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    rp = install_path(net, RoutingPathMulti("n1", "n4", swap=[1, 0, 1]))

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
    print_fw_counters(net)

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
def test_tree2_dynepr(t_edge_etg: float, selected_path: tuple[int, int], n_consumed: tuple[int, int]):
    """Test MuxSchemeDynamicEpr in tree (height=2) topology."""

    def select_path(epr: Entanglement, fib: Fib, path_ids: list[int]) -> int:
        _ = fib
        if len(path_ids) != 2:
            chosen = path_ids[0]
        elif epr.src is f2.node:  # n2-n1
            chosen = (rp0.path_id, rp1.path_id)[selected_path[0]]
        else:  # n1-n3
            assert epr.src is f1.node
            chosen = (rp0.path_id, rp1.path_id)[selected_path[1]]
        return chosen

    net, simulator = build_tree_network(ps=1.0, mux=MuxSchemeDynamicEpr(select_path=select_path))
    f1, f2, f3, f4, f5, f6, f7 = (node.get_app(ProactiveForwarder) for node in net.nodes)

    # n4-n2-n1-n3-n6
    rp0 = install_path(net, RoutingPathSingle("n4", "n6", qubit_allocation=QubitAllocationType.DISABLED, swap="asap"))
    # n5-n2-n1-n3-n7
    rp1 = install_path(net, RoutingPathSingle("n5", "n7", qubit_allocation=QubitAllocationType.DISABLED, swap="asap"))

    provide_entanglements(
        (t_edge_etg, f4, f2),
        (t_edge_etg, f5, f2),
        (t_edge_etg, f3, f6),
        (t_edge_etg, f3, f7),
        (1.003, f2, f1),
        (1.005, f1, f3),
    )
    simulator.run()
    print_fw_counters(net)

    assert f4.cnt.n_consumed == n_consumed[0] == f6.cnt.n_consumed
    assert f5.cnt.n_consumed == n_consumed[1] == f7.cnt.n_consumed


@pytest.mark.parametrize(
    ("t_edge_etg", "selected_qubit", "selected_path_1", "n_consumed"),
    [
        # 1. Edge entanglements arrive first.
        # 2. Center entanglements arrive next:
        #    n2 chooses n4-n2 on path 0 to swap with n2-n1.
        #    n3 chooses n3-n6 on path 0 to swap with n1-n3.
        #    n1 can choose either FIB entry without affecting outcome.
        # 3. Path 0 should get end-to-end entanglement.
        ((1.001, 1.001, 1.001, 1.001), (0, 0), 0, (1, 0)),
        ((1.001, 1.001, 1.001, 1.001), (0, 0), 1, (1, 0)),
        # 1. Edge entanglements arrive first.
        # 2. Center entanglements arrive next:
        #    n2 chooses n4-n2 on path 0 to swap with n2-n1.
        #    n3 chooses n3-n6 on path 1 to swap with n1-n3.
        #    n1 can choose either FIB entry without affecting outcome.
        # 3. Neither path could get end-to-end entanglement.
        ((1.001, 1.001, 1.001, 1.001), (0, 1), 0, (0, 0)),
        ((1.001, 1.001, 1.001, 1.001), (0, 1), 1, (0, 0)),
        # 1. Center entanglements arrive first.
        # 2. Path 1 edge entanglements arrive next:
        #    n2 and n3 each has only one matching center entanglement to swap with edge entanglement;
        #    each has only one matching FIB entry.
        #    n1 can choose either FIB entry without affecting outcome.
        # 3. Path 0 edge entanglements arrive last, but neither n2 nor n3 can swap them.
        # 4. Path 1 should get end-to-end entanglement.
        ((1.009, 1.007, 1.009, 1.007), (9, 9), 0, (0, 1)),
        ((1.009, 1.007, 1.009, 1.007), (9, 9), 1, (0, 1)),
        # 1. Center entanglements arrive first.
        # 2. n4-n2 on path 0 and n3-n7 on path 1 edge entanglements arrive next:
        #    n2 and n3 each has only one matching center entanglement to swap with edge entanglement.
        #    n1 can choose either FIB entry without affecting outcome.
        # 3. n5-n2 on path 1 and n3-n6 on path 0 edge entanglements arrive last, but neither n2 nor n3 can swap them.
        # 4. Neither path could get end-to-end entanglement.
        ((1.009, 1.007, 1.007, 1.009), (9, 9), 0, (0, 0)),
        ((1.009, 1.007, 1.007, 1.009), (9, 9), 1, (0, 0)),
        # `9` means the selection callback shouldn't have been invoked with two candidates.
    ],
)
def test_tree2_statistical(
    t_edge_etg: tuple[float, float, float, float],
    selected_qubit: tuple[int, int],
    selected_path_1: int,
    n_consumed: tuple[int, int],
):
    """Test MuxSchemeStatistical in tree (height=2) topology."""

    def select_qubit(fw: Forwarder, mt0: MemoryEprTuple, candidates: list[MemoryEprTuple]) -> MemoryEprTuple:
        _ = mt0
        if len(candidates) != 2:
            chosen = candidates[0]
        elif fw is f2:  # n2-n1 choosing between n4-n2 and n5-n2
            partner = (f4, f5)[selected_qubit[0]]
            chosen = next((mt1 for mt1 in candidates if mt1[1].src is partner.node))
        elif fw is f3:  # n1-n3 choosing between n3-n6 and n3-n7
            partner = (f6, f7)[selected_qubit[1]]
            chosen = next((mt1 for mt1 in candidates if mt1[1].dst is partner.node))
        else:
            raise RuntimeError()
        return chosen

    def select_path(fw: Forwarder, epr0: Entanglement, epr1: Entanglement, path_ids: list[int]) -> int:
        _ = epr0, epr1
        if len(path_ids) != 2:
            chosen = path_ids[0]
        elif fw is f1:  # n1 choosing for n2-n1-n3 swap
            chosen = (rp0.path_id, rp1.path_id)[selected_path_1]
        else:
            # In all other nodes, only one FIB entry is matched after choosing qubit.
            raise RuntimeError()
        return chosen

    net, simulator = build_tree_network(
        ps=1.0, mux=MuxSchemeStatistical(select_swap_qubit=select_qubit, select_path=select_path)
    )
    f1, f2, f3, f4, f5, f6, f7 = (node.get_app(ProactiveForwarder) for node in net.nodes)

    # n4-n2-n1-n3-n6
    rp0 = install_path(net, RoutingPathSingle("n4", "n6", qubit_allocation=QubitAllocationType.DISABLED, swap="asap"))
    # n5-n2-n1-n3-n7
    rp1 = install_path(net, RoutingPathSingle("n5", "n7", qubit_allocation=QubitAllocationType.DISABLED, swap="asap"))

    provide_entanglements(
        (t_edge_etg[0], f4, f2),
        (t_edge_etg[1], f5, f2),
        (t_edge_etg[2], f3, f6),
        (t_edge_etg[3], f3, f7),
        (1.003, f2, f1),
        (1.005, f1, f3),
    )
    simulator.run()
    print_fw_counters(net)

    assert f4.cnt.n_consumed == n_consumed[0] == f6.cnt.n_consumed
    assert f5.cnt.n_consumed == n_consumed[1] == f7.cnt.n_consumed
    assert f2.cnt.n_swap_conflict + f1.cnt.n_swap_conflict + f3.cnt.n_swap_conflict == (2 if sum(n_consumed) == 0 else 0)


@pytest.mark.parametrize(
    ("etg_sec", "n_cutoff"),
    [
        # n1-n2 arrives at t=1.005, n2-n3 arrives at t=1.006, swapped.
        ((1.004, 1.005), (0, 0)),
        ((1.005, 1.004), (0, 0)),
        # n1-n2 arrives at t=1.006 and is discarded at t=1.008.
        # n2-n3 arrives at t=1.009.
        ((1.005, 1.008), (1, 0)),
        ((1.008, 1.005), (0, 1)),
        # n1-n2 arrives at t=1.003 and is discarded at t=1.005.
        # n2-n3 arrives at t=1.006 and is discarded at t=1.008.
        ((1.002, 1.005), (1, 1)),
        ((1.005, 1.002), (1, 1)),
    ],
)
def test_3_waittime(etg_sec: tuple[float, float], n_cutoff: tuple[int, int]):
    """Test CutoffSchemeWaitTime in 3-node topology."""
    net, simulator = build_linear_network(3, ps=1.0, end_time=1.010)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(net, RoutingPathSingle("n1", "n3", swap=[1, 0, 1], swap_cutoff=[0, 0.002, 0]))
    provide_entanglements(
        (etg_sec[0], f1, f2),
        (etg_sec[1], f2, f3),
    )
    simulator.run()
    print_fw_counters(net)

    assert f1.cnt.n_cutoff == [0, n_cutoff[0]]
    assert f2.cnt.n_cutoff == [sum(n_cutoff), 0]
    assert f3.cnt.n_cutoff == [0, n_cutoff[1]]
    n_swapped = 1 - max(n_cutoff)
    assert f2.cnt.n_swapped == n_swapped
    assert f1.cnt.n_consumed == n_swapped == f3.cnt.n_consumed
