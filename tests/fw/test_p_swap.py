"""
Test suite for ProactiveForwarder focused on swapping.
"""

import itertools
from collections.abc import Sequence

import pytest

from mqns.entity.timer import Timer
from mqns.models.delay import ConstantDelayModel
from mqns.models.epr import Entanglement, MixedStateEntanglement
from mqns.models.error import PerfectErrorModel
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

from .fw_common import (
    QubitReleaseLoggerApp,
    build_linear_network,
    build_rect_network,
    build_tree_network,
    check_fw_counters,
    check_memory_released,
    collect_cpacket_counts,
    install_path,
    print_fw_counters,
    provide_entanglements,
)


def test_3_disabled():
    """Test swap disabled mode."""
    net, simulator = build_linear_network(3, fw={"p_swap": 1.0})
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
    check_fw_counters(
        net,
        n_consumed=(1, 2, 1),
        n_swapped=(0, 0, 0),
    )


@pytest.mark.parametrize(
    ("swap_delay", "n_consumed"),
    [
        # 1. t=1.0010, elementary EPRs arrive, n2 starts swapping.
        # 2. t=1.0012, n2 completes swapping, heralds success to n1+n3.
        # 3. t=1.0017, n1/n3 receives heralding, consumes EPR.
        # 4. t=1.0020, memory qubits would decohere, but it's already consumed.
        (0.0002, 1),
        # 1. t=1.0010, elementary EPRs arrive, n2 starts swapping.
        # 2. t=1.0017, n2 completes swapping, heralds success to n1+n3.
        # 3. t=1.0020, memory qubits decohere.
        # 4. t=1.0016, n1/n3 receives heralding, ignores due to qubit not in memory.
        (0.0007, 0),
        # 1. t=1.0010, elementary EPRs arrive, n2 starts swapping.
        # 2. t=1.0020, memory qubits decohere.
        # 2. t=1.0022, n2 aborts swapping, heralds failure to n1+n3.
        # 3. t=1.0027, n1/n3 receives heralding, ignores due to qubit not in memory.
        (0.0012, 0),
    ],
)
def test_3_decohere(swap_delay: float, n_consumed: int):
    """Test short decoherence time in 3-node topology."""
    net, simulator = build_linear_network(3, t_cohere=0.002, fw={"p_swap": 1.0, "swap_delay": swap_delay}, end_time=2)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(net, RoutingPathSingle("n1", "n3", swap=[1, 0, 1]))
    provide_entanglements(
        (1, f1, f2),
        (1, f2, f3),
    )
    simulator.run()
    print_fw_counters(net)
    check_fw_counters(
        net,
        n_consumed=(n_consumed, 0, n_consumed),
        n_su_lower=(1, 0, 1),
    )


@pytest.mark.parametrize(
    ("etg_sec", "swap_delay", "n_consumed", "n_cutoff"),
    [
        # 1. t=1.0050, n1-n2 arrives, discard scheduled at 1.007.
        # 2. t=1.0060, n2-n3 arrives, n1-n2 discard canceled, n2 starts swapping.
        # 3. t=1.0065, n2 finishes swapping, heralds success to n1+n3.
        # 4. t=1.0070, n1/n3 consumes EPR.
        ((1.004, 1.005), 0.0005, 1, (0, 0)),
        # 1. t=1.0050, n1-n2 arrives, discard scheduled at 1.007.
        # 2. t=1.0060, n2-n3 arrives, n1-n2 discard canceled, n2 starts swapping.
        # 3. t=1.0072, n2 finishes swapping, heralds success to n1+n3.
        # 4. t=1.0077, n1/n3 consumes EPR.
        ((1.004, 1.005), 0.0012, 1, (0, 0)),
        # 1. t=1.0050, n1-n2 arrives, discard scheduled at 1.007.
        # 2. t=1.0060, n2-n3 arrives, n1-n2 discard canceled, n2 starts swapping.
        # 3. t=1.0080, n1-n2 decoheres.
        # 3. t=1.0085, n2 aborts swapping, heralds failure to n1+n3.
        ((1.004, 1.005), 0.0025, 0, (0, 0)),
        # 1. t=1.0060, n1-n2 arrives, discard scheduled at 1.007.
        # 2. t=1.0080, n1-n2 is discarded.
        # 3. t=1.0090, n2-n3 arrives, discard scheduled at 1.011.
        ((1.005, 1.008), 0.0005, 0, (1, 0)),
        # 1. t=1.0030, n1-n2 arrives, discard scheduled at 1.007.
        # 2. t=1.0050, n1-n2 is discarded.
        # 3. t=1.0060, n2-n3 arrives, discard scheduled at 1.008.
        # 3. t=1.0080, n2-n3 is discarded.
        ((1.002, 1.005), 0.0005, 0, (1, 1)),
    ],
)
def test_3_waittime(etg_sec: tuple[float, float], swap_delay: float, n_consumed: int, n_cutoff: tuple[int, int]):
    """Test CutoffSchemeWaitTime in 3-node topology."""
    net, simulator = build_linear_network(3, t_cohere=0.004, fw={"p_swap": 1.0, "swap_delay": swap_delay}, end_time=1.010)
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

    check_fw_counters(
        net,
        n_consumed=(n_consumed, 0, n_consumed),
        n_swapped=(0, n_consumed, 0),
    )
    assert f1.cnt.n_cutoff == [0, n_cutoff[0]]
    assert f2.cnt.n_cutoff == [sum(n_cutoff), 0]
    assert f3.cnt.n_cutoff == [0, n_cutoff[1]]


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
    net, simulator = build_linear_network(4, fw={"p_swap": 1.0}, timing=timing)
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
    ("etg_ms", "ps3"),
    itertools.product(
        [
            (1, 2, 1),
            (2, 1, 2),
            (1, 2, 3),
            (3, 2, 1),
        ],
        (1, 0),
    ),
)
def test_4_asap(etg_ms: tuple[int, int, int], ps3: int):
    """Test SWAP-ASAP in 4-node topology with various entanglement arrival orders."""
    net, simulator = build_linear_network(4, fw={"p_swap": 1.0}, end_time=2)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    f3.swap.ps = ps3
    install_path(net, RoutingPathSingle("n1", "n4", swap=[1, 0, 0, 1]))
    provide_entanglements(
        (1 + etg_ms[0] / 1000, f1, f2),
        (1 + etg_ms[1] / 1000, f2, f3),
        (1 + etg_ms[2] / 1000, f3, f4),
    )
    simulator.run()
    print_fw_counters(net)
    if ps3 == 1:
        check_fw_counters(
            net,
            n_consumed=(1, 0, 0, 1),
            n_swapped=(0, 1, 1, 0),
            n_swap_fail=(0, 0, 0, 0),
            n_su_same=(0, 1, 1, 0),
            n_su_lower=(1, 0, 0, 1),
        )
    else:
        check_fw_counters(
            net,
            n_consumed=(0, 0, 0, 0),
            n_swapped=(0, 1, 0, 0),
            n_swap_fail=(0, 0, 1, 0),
            n_su_same=(0, 1, (0, 1), 0),  # if n2 hears n3's failure before its own swap, it does not herald n3
            n_su_lower=(1, 0, 0, 1),
        )
    check_memory_released(net)


@pytest.mark.parametrize(
    ("ps3", "delay3", "n_swap2", "n_consumed", "t_release", "n_cpacket"),
    [
        # 1. t=1.0110, n2-n3 arrives, both n2 and n3 start swapping.
        # 2. t=1.0110, n3 completes swapping with success, heralds n2 for n2-n4.
        # 3. t=1.0115, n2 receives n2-n4 heralding.
        # 4. t=1.0610, n2 completes swapping with success, heralds n1 for n1-n4, heralds n3 for n1-n3.
        # 5. t=1.0615, n1 receives n1-n4 heralding, consumes EPR.
        # 6. t=1.0615, n3 receives n1-n3 heralding, heralds n4 for n1-n4.
        # 7. t=1.0620, n4 receives n1-n4 heralding, consumes EPR.
        (1.0, 0.0000, 1, 1, (1.0615, 1.0610, 1.0110, 1.0620), (1, 1, 1, 1)),
        # 1. t=1.0110, n2-n3 arrives, both n2 and n3 start swapping.
        # 2. t=1.0110, n3 completes swapping with failure, heralds n2+n4 for failure.
        # 3. t=1.0115, n4 receives failure heralding, releases qubit.
        # 4. t=1.0115, n2 receives failure heralding, heralds n1 for failure.
        # 5. t=1.0120, n1 receives failure heralding, releases qubit.
        # 4. t=1.0610, n2 completes swapping with success, ignores due to earlier failure.
        (0.0, 0.0000, 1, 0, (1.0120, 1.0610, 1.0110, 1.0115), (1, 1, 0, 1)),
        # 1. t=1.0110, n2-n3 arrives, both n2 and n3 start swapping.
        # 2. t=1.0609, n3 completes swapping with success, heralds n2 for n2-n4.
        # 3. t=1.0610, n2 completes swapping with success, heralds n3 for n1-n3.
        # 4. t=1.0614, n2 receives n2-n4 heralding, heralds n1 for n1-n4.
        # 5. t=1.0615, n3 receives n1-n3 heralding, heralds n4 for n1-n4.
        # 6. t=1.0619, n1 receives n1-n4 heralding, consumes EPR.
        # 7. t=1.0620, n4 receives n1-n4 heralding, consumes EPR.
        (1.0, 0.0499, 1, 1, (1.0619, 1.0610, 1.0609, 1.0620), (1, 1, 1, 1)),
        # 1. t=1.0110, n2-n3 arrives, both n2 and n3 start swapping.
        # 2. t=1.0609, n3 completes swapping with failure, heralds n2+n4 for failure.
        # 3. t=1.0610, n2 completes swapping with success, heralds n3 for n1-n3.
        # 4. t=1.0614, n4 receives failure heralding, releases qubit.
        # 5. t=1.0614, n2 receives failure heralding, heralds n1 for failure.
        # 6. t=1.0615, n3 receives n1-n3 heralding, ignores due to earlier failure.
        # 6. t=1.0619, n1 receives failure heralding, releases EPR.
        (0.0, 0.0499, 1, 0, (1.0619, 1.0610, 1.0609, 1.0614), (1, 1, 1, 1)),
        # 1. t=1.0110, n2-n3 arrives, both n2 and n3 start swapping.
        # 2. t=1.0610, n2 completes swapping with success, heralds n3 for n1-n3.
        # 3. t=1.0611, n3 completes swapping with success, heralds n2 for n2-n4.
        # 4. t=1.0615, n3 receives n1-n3 heralding, heralds n4 for n1-n4.
        # 5. t=1.0616, n2 receives n2-n4 heralding, heralds n1 for n1-n4.
        # 6. t=1.0620, n4 receives n1-n4 heralding, consumes EPR.
        # 7. t=1.0621, n1 receives n1-n4 heralding, consumes EPR.
        (1.0, 0.0501, 1, 1, (1.0621, 1.0610, 1.0611, 1.0620), (1, 1, 1, 1)),
        # 1. t=1.0110, n2-n3 arrives, both n2 and n3 start swapping.
        # 2. t=1.0610, n2 completes swapping with success, heralds n3 for n1-n3.
        # 3. t=1.0611, n3 completes swapping with failure, heralds n2+n4 for failure.
        # 4. t=1.0615, n3 receives n1-n3 heralding, ignores due to earlier failure.
        # 5. t=1.0616, n4 receives failure heralding, releases qubit.
        # 6. t=1.0616, n2 receives failure heralding, heralds n1 for failure.
        # 7. t=1.0621, n1 receives failure heralding, releases qubit.
        (0.0, 0.0501, 1, 0, (1.0621, 1.0610, 1.0611, 1.0616), (1, 1, 1, 1)),
    ],
)
def test_4_delayed(
    monkeypatch: pytest.MonkeyPatch,
    ps3: float,
    delay3: float,
    n_swap2: int,
    n_consumed: int,
    t_release: tuple[float, float],
    n_cpacket: tuple[int, int, int, int],
):
    """Test swap delay model and error model in 4-node topology."""
    net, simulator = build_linear_network(
        4,
        epr_type=MixedStateEntanglement,
        fw={
            "p_swap": 1.0,
            "swap_delay": 0.050,
            "swap_error": "DEPOLAR:0.3",
        },
        end_time=2,
    )
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    f3.swap.ps = ps3
    f3.swap.delay = ConstantDelayModel(delay3)
    f3.swap.error = PerfectErrorModel()

    f2_n_swapped_values: list[int] = []

    def save_counter():
        f2_n_swapped_values.append(f2.cnt.n_swapped)

    timer = Timer("save_counters", start_time=1.018, end_time=1.088, step_time=0.010, trigger_func=save_counter)
    timer.install(simulator)

    install_path(net, RoutingPathSingle("n1", "n4", swap=[1, 0, 0, 1]))
    provide_entanglements(
        (1.000, f1, f2),
        (1.000, f3, f4),
        (1.010, f2, f3),
        fidelity=1,
    )
    cpacket_cnt = collect_cpacket_counts(monkeypatch)
    simulator.run()
    print_fw_counters(net)
    check_memory_released(net)
    print("cpacket_cnt", cpacket_cnt)

    assert f2_n_swapped_values == [0, 0, 0, 0, 0, n_swap2, n_swap2, n_swap2]
    assert f1.cnt.n_consumed == n_consumed
    if n_consumed > 0:
        assert 0.5 < f1.cnt.consumed_avg_fidelity <= 0.75
        assert f1.cnt.consumed_avg_fidelity == pytest.approx(f4.cnt.consumed_avg_fidelity)

    assert list(fw.node.get_app(QubitReleaseLoggerApp).last_time for fw in (f1, f2, f3, f4)) == [
        simulator.time(sec=t) for t in t_release
    ]
    assert (cpacket_cnt["*-n1"], cpacket_cnt["*-n2"], cpacket_cnt["*-n3"], cpacket_cnt["*-n4"]) == n_cpacket


@pytest.mark.parametrize(
    ("swap_delay", "n_consumed"),
    [
        # 1. t=1.0010, elementary EPRs arrive, n2 starts swapping.
        # 2. t=1.0012, n2/n3 completes swapping, heralds success to n3/n2.
        # 3. t=1.0017, n3/n2 receives heralding, heralds success to n4/n1.
        # 4. t=1.0022, n4/n1 receives heralding, consumes EPR.
        # 5. t=1.0030, memory qubits would decohere, but it's already consumed.
        (0.0002, 1),
        # 1. t=1.0010, elementary EPRs arrive, n2 starts swapping.
        # 2. t=1.0022, n2/n3 completes swapping, heralds success to n3/n2.
        # 3. t=1.0027, n3/n2 receives heralding, heralds success to n4/n1.
        # 4. t=1.0030, memory qubits decohere.
        # 5. t=1.0032, n4/n1 receives heralding, ignores due to qubit not in memory.
        (0.0012, 0),
        # 1. t=1.0010, elementary EPRs arrive, n2 starts swapping.
        # 2. t=1.0027, n2/n3 completes swapping, heralds success to n3/n2.
        # 3. t=1.0030, memory qubits decohere.
        # 4. t=1.0032, n3/n2 receives heralding, heralds success to n4/n1.
        # 5. t=1.0037, n4/n1 receives heralding, ignores due to qubit not in memory.
        (0.0017, 0),
        # 1. t=1.0010, elementary EPRs arrive, n2 starts swapping.
        # 2. t=1.0030, memory qubits decohere.
        # 3. t=1.0032, n2/n3 aborts swapping, heralds failure to n3+n1/n2+n4.
        # 4. t=1.0037, n3/n2 receives heralding, ignores due to earlier failure.
        # 5. t=1.0037, n1/n4 receives heralding, ignores due to qubit not in memory.
        (0.0022, 0),
    ],
)
def test_4_decohere(swap_delay: float, n_consumed: int):
    """Test short decoherence time in 4-node topology."""
    net, simulator = build_linear_network(4, t_cohere=0.003, fw={"p_swap": 1.0, "swap_delay": swap_delay}, end_time=2)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    install_path(net, RoutingPathSingle("n1", "n4", swap=[1, 0, 0, 1]))
    provide_entanglements(
        (1, f1, f2),
        (1, f2, f3),
        (1, f3, f4),
    )
    simulator.run()
    print_fw_counters(net)
    check_fw_counters(
        net,
        n_consumed=(n_consumed, 0, 0, n_consumed),
        n_su_lower=(1, 0, 0, 1),
    )


@pytest.mark.parametrize(("ps3", "etg_ms"), itertools.product((1, 0), ((2, 1, 1, 2), (1, 2, 2, 1))))
def test_5_asap(
    ps3: float,
    etg_ms: tuple[int, int, int, int],
):
    """Test SWAP-ASAP in 5-node topology with various entanglement arrival orders."""
    n_consumed = 1 if ps3 == 1 else 0

    net, simulator = build_linear_network(5, fw={"p_swap": 1.0}, end_time=2)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)
    f3.swap.ps = ps3

    install_path(net, RoutingPathSingle("n1", "n5", swap="asap"))
    provide_entanglements(
        (1 + etg_ms[0] / 1000, f1, f2),
        (1 + etg_ms[1] / 1000, f2, f3),
        (1 + etg_ms[2] / 1000, f3, f4),
        (1 + etg_ms[3] / 1000, f4, f5),
    )
    simulator.run()
    print_fw_counters(net)
    check_fw_counters(
        net,
        n_consumed=(n_consumed, 0, 0, 0, n_consumed),
        n_swapped=(0, 1, n_consumed, 1, 0),
        n_swap_fail=(0, 0, 1 - n_consumed, 0, 0),
        n_su_same=(0, 1, (0, 1, 2), 1, 0),
        n_su_lower=(1, 0, 0, 0, 1),
    )
    check_memory_released(net)


@pytest.mark.parametrize(
    ("swap_sulower", "etg_ms"),
    itertools.product(
        [
            ((3, 0, 1, 2, 3), (1, 0, 1, 1, 1)),  # l2r
            ((3, 2, 1, 0, 3), (1, 1, 1, 0, 1)),  # r2l
            ((3, 0, 1, 0, 3), (1, 0, 2, 0, 1)),  # baln
        ],
        itertools.permutations(range(4), 4),
    ),
)
def test_5_sequential(swap_sulower: tuple[Sequence[int], Sequence[int]], etg_ms: tuple[int, int, int, int]):
    """Test sequential swap orders with various entanglement arrival orders."""
    swap, su_lower = swap_sulower

    net, simulator = build_linear_network(5, fw={"p_swap": 1.0})
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
    check_fw_counters(
        net,
        n_consumed=(1, 0, 0, 0, 1),
        n_swapped=(0, 1, 1, 1, 0),
        n_swap_fail=(0, 0, 0, 0, 0),
        n_su_same=(0, 0, 0, 0, 0),
        n_su_lower=su_lower,
    )


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
    net, simulator = build_rect_network(fw={"p_swap": 1.0})
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

    net, simulator = build_tree_network(
        fw={"p_swap": 1.0, "mux": MuxSchemeDynamicEpr(select_path=select_path)},
        # If there is a conflict in path selection, there will be two leftover EPRs.
        swap_table_leak_tol=2 if selected_path[0] != selected_path[1] else 0,
    )
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
    ("etgs", "selected_qubit", "selected_path", "n_consumed"),
    [
        # EPRs arrive in left-to-right order: n4-n2 & n5-n2, n2-n1, n1-n3, n3-n6 & n3-n7.
        # n3, n1, n2 perform their swaps sequentially.
        ((1, 1, 2, 3, 4, 5), (0, 9), -1, (1, 0)),
        ((1, 1, 2, 3, 5, 4), (0, 9), -1, (1, 0)),
        ((1, 1, 2, 3, 4, 5), (1, 9), -1, (0, 1)),
        ((1, 1, 2, 3, 5, 4), (1, 9), -1, (0, 1)),
        # EPRs arrive in outer-to-inner order: n4-n2 / n5-n2, n3-n6 / n3-n7, n2-n1 & n1-n3.
        # n2 and n3 swap first in parallel (without choice), and n1 swaps last.
        ((1, -1, 2, 2, 1, -1), (9, 9), -1, (1, 0)),
        ((-1, 1, 2, 2, -1, 1), (9, 9), -1, (0, 1)),
        ((1, -1, 2, 2, -1, 1), (9, 9), -1, (0, 0)),
        # EPRs arrive in inner-to-outer order: n2-n1 & n1-n3, n4-n2 / n5-n2, n3-n6 / n3-n7.
        # n1 swaps first, n2 and n3 swap last in parallel.
        ((2, -1, 1, 1, 2, -1), (9, 9), -1, (1, 0)),
        ((-1, 2, 1, 1, -1, 2), (9, 9), -1, (0, 1)),
        ((2, -1, 1, 1, -1, 2), (9, 9), -1, (0, 0)),
        # EPRs arrive in inner-to-outer order: n2-n1 & n1-n3, n4-n2 & n5-n2, n3-n6 & n3-n7.
        # n1 swaps first, n2 and n3 swap last in parallel but with physically unrealistic coordinated decisions.
        ((2, 3, 1, 1, 3, 2), (9, 9), 0, (1, 0)),
        ((2, 3, 1, 1, 3, 2), (9, 9), 1, (0, 1)),
    ],
)
def test_tree2_statistical(
    etgs: tuple[int, int, int, int, int, int], selected_qubit: tuple[int, int], selected_path: int, n_consumed: tuple[int, int]
):
    """Test MuxSchemeStatistical in tree (height=2) topology."""

    def select_qubit(candidates: list[MemoryEprTuple], fw: Forwarder, mt0: MemoryEprTuple) -> MemoryEprTuple:
        _ = mt0
        assert len(candidates) == 2
        if fw is f2:  # n2-n1 choosing between n4-n2 and n5-n2
            partner = (f4, f5)[selected_qubit[0]]
            chosen = next((mt1 for mt1 in candidates if mt1[1].src is partner.node), None)
        elif fw is f3:  # n1-n3 choosing between n3-n6 and n3-n7
            partner = (f6, f7)[selected_qubit[1]]
            chosen = next((mt1 for mt1 in candidates if mt1[1].dst is partner.node), None)
        else:
            raise RuntimeError()
        assert chosen is not None
        return chosen

    def select_path(candidates: list[int], fw: Forwarder, epr0: Entanglement, epr1: Entanglement) -> int:
        _ = fw, epr0, epr1
        assert len(candidates) == 2
        return selected_path

    if selected_path < 0:
        mux = MuxSchemeStatistical(select_swap_qubit=select_qubit)
    else:
        mux = MuxSchemeStatistical(select_swap_qubit=select_qubit, coordinated_decisions=True, select_path=select_path)

    net, simulator = build_tree_network(
        fw={"p_swap": 1.0, "mux": mux},
        end_time=2,
    )
    f1, f2, f3, f4, f5, f6, f7 = (node.get_app(ProactiveForwarder) for node in net.nodes)

    # n4-n2-n1-n3-n6
    install_path(net, RoutingPathSingle("n4", "n6", qubit_allocation=QubitAllocationType.DISABLED, swap="asap"))
    # n5-n2-n1-n3-n7
    install_path(net, RoutingPathSingle("n5", "n7", qubit_allocation=QubitAllocationType.DISABLED, swap="asap"))

    edges = ((f4, f2), (f5, f2), (f2, f1), (f1, f3), (f3, f6), (f3, f7))
    provide_entanglements(*(((1 + 0.001 * t) if t >= 0 else -1, *edge) for (t, edge) in zip(etgs, edges)))
    simulator.run()
    print_fw_counters(net)
    if -1 in etgs:
        # For test cases without unused EPRs, swap conflict should release memory quickly.
        check_memory_released(net)

    assert f4.cnt.n_consumed == n_consumed[0] == f6.cnt.n_consumed
    assert f5.cnt.n_consumed == n_consumed[1] == f7.cnt.n_consumed
    assert sum(fw.cnt.n_su_lower[4] for fw in (f4, f5, f6, f7)) == (2 if sum(n_consumed) == 0 else 0)
