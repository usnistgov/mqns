"""
Test suite for ProactiveForwarder focused on swapping.
"""

import itertools
from collections.abc import Sequence
from typing import cast

import pytest

from mqns.entity.node import QNode
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
    RoutingPathStatic,
)
from mqns.network.network import TimingModeSync
from mqns.network.proactive import ProactiveForwarder
from mqns.simulator import func_to_event

from .fw_common import (
    QubitReleaseReset,
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
    fwA, fwB, fwC = (node.get_app(ProactiveForwarder) for node in net.nodes)

    rp = install_path(net, RoutingPathStatic("ABC", swap=[0, 0, 0]))

    def check_fib_entries():
        for fw in (fwA, fwB, fwC):
            fib_entry = fw.fib.get(rp.path_id)
            assert fib_entry.is_swap_disabled is True

    simulator.add_event(func_to_event(simulator.time(sec=2.0), check_fib_entries))
    provide_entanglements(
        (1.001, fwA, fwB),
        (1.002, fwB, fwC),
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
        # 1. t=1.0010, elementary EPRs arrive, B starts swapping.
        # 2. t=1.0012, B completes swapping, heralds success to A+C.
        # 3. t=1.0017, A/C receives heralding, consumes EPR.
        # 4. t=1.0020, memory qubits would decohere, but it's already consumed.
        (0.0002, 1),
        # 1. t=1.0010, elementary EPRs arrive, B starts swapping.
        # 2. t=1.0017, B completes swapping, heralds success to A+C.
        # 3. t=1.0020, memory qubits decohere.
        # 4. t=1.0016, A/C receives heralding, ignores due to qubit not in memory.
        (0.0007, 0),
        # 1. t=1.0010, elementary EPRs arrive, B starts swapping.
        # 2. t=1.0020, memory qubits decohere.
        # 2. t=1.0022, B aborts swapping, heralds failure to A+C.
        # 3. t=1.0027, A/C receives heralding, ignores due to qubit not in memory.
        (0.0012, 0),
    ],
)
def test_3_decohere(swap_delay: float, n_consumed: int):
    """Test short decoherence time in 3-node topology."""
    net, simulator = build_linear_network(3, t_cohere=0.002, fw={"p_swap": 1.0, "swap_delay": swap_delay}, end_time=2)
    fwA, fwB, fwC = (node.get_app(ProactiveForwarder) for node in net.nodes)

    install_path(net, RoutingPathStatic("ABC"))
    provide_entanglements(
        (1, fwA, fwB),
        (1, fwB, fwC),
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
        # 1. t=1.0050, A-B arrives, discard scheduled at 1.007.
        # 2. t=1.0060, B-C arrives, A-B discard canceled, B starts swapping.
        # 3. t=1.0065, B finishes swapping, heralds success to A+C.
        # 4. t=1.0070, A/C consumes EPR.
        ((1.004, 1.005), 0.0005, 1, (0, 0)),
        # 1. t=1.0050, A-B arrives, discard scheduled at 1.007.
        # 2. t=1.0060, B-C arrives, A-B discard canceled, B starts swapping.
        # 3. t=1.0072, B finishes swapping, heralds success to A+C.
        # 4. t=1.0077, A/C consumes EPR.
        ((1.004, 1.005), 0.0012, 1, (0, 0)),
        # 1. t=1.0050, A-B arrives, discard scheduled at 1.007.
        # 2. t=1.0060, B-C arrives, A-B discard canceled, B starts swapping.
        # 3. t=1.0080, A-B decoheres.
        # 3. t=1.0085, B aborts swapping, heralds failure to A+C.
        ((1.004, 1.005), 0.0025, 0, (0, 0)),
        # 1. t=1.0060, A-B arrives, discard scheduled at 1.007.
        # 2. t=1.0080, A-B is discarded.
        # 3. t=1.0090, B-C arrives, discard scheduled at 1.011.
        ((1.005, 1.008), 0.0005, 0, (1, 0)),
        # 1. t=1.0030, A-B arrives, discard scheduled at 1.007.
        # 2. t=1.0050, A-B is discarded.
        # 3. t=1.0060, B-C arrives, discard scheduled at 1.008.
        # 3. t=1.0080, B-C is discarded.
        ((1.002, 1.005), 0.0005, 0, (1, 1)),
    ],
)
def test_3_waittime(etg_sec: tuple[float, float], swap_delay: float, n_consumed: int, n_cutoff: tuple[int, int]):
    """Test CutoffSchemeWaitTime in 3-node topology."""
    net, simulator = build_linear_network(3, t_cohere=0.004, fw={"p_swap": 1.0, "swap_delay": swap_delay}, end_time=1.010)
    fwA, fwB, fwC = (node.get_app(ProactiveForwarder) for node in net.nodes)

    install_path(net, RoutingPathStatic("ABC", swap_cutoff=[0, 0.002, 0]))
    provide_entanglements(
        (etg_sec[0], fwA, fwB),
        (etg_sec[1], fwB, fwC),
    )
    simulator.run()
    print_fw_counters(net)

    check_fw_counters(
        net,
        n_consumed=(n_consumed, 0, n_consumed),
        n_swapped=(0, n_consumed, 0),
    )
    assert fwA.cnt.n_cutoff == [0, n_cutoff[0]]
    assert fwB.cnt.n_cutoff == [sum(n_cutoff), 0]
    assert fwC.cnt.n_cutoff == [0, n_cutoff[1]]


@pytest.mark.parametrize(
    ("t_ext", "expected"),
    [
        # 1. Elementary entanglements arrive during EXTERNAL phase.
        # 2. A-B-C is swapped at t=0.008400s when INTERNAL phase begins.
        # 3. A and C are informed at t=0.008900s.
        # 4. A-C-D is swapped at t=0.008900s.
        # 5. D is informed at t=0.009400s and consumes the end-to-end entanglement.
        # 6. A is informed at t=0.009900s and consumes the end-to-end entanglement.
        (0.008400, (1, 1, 1, 1)),
        # 1. Elementary entanglements arrive during EXTERNAL phase.
        # 2. A-B-C is swapped at t=0.008900s when INTERNAL phase begins.
        # 3. A and C are informed at 0=1.009400s.
        # 4. A-C-D is swapped at t=0.009400s.
        # 5. D is informed at t=0.009900s and consumes the end-to-end entanglement.
        # 6. A is informed at t=0.010400s but INTERNAL phase has ended.
        (0.008900, (0, 1, 1, 1)),
        # 1. Elementary entanglements arrive during EXTERNAL phase.
        # 2. A-B-C is swapped at t=0.009400s when INTERNAL phase begins.
        # 3. A and C are informed at t=0.009900s.
        # 4. A-C-D is swapped at t=0.009900s.
        # 5. D is informed at t=0.010400s but INTERNAL phase has ended.
        # 6. A is informed at t=0.010900s but INTERNAL phase has ended.
        (0.009400, (0, 1, 1, 0)),
        # 1. Elementary entanglements arrive during EXTERNAL phase.
        # 2. A-B-C is swapped at t=0.009900s when INTERNAL phase begins.
        # 3. A and C are informed at t=0.010400s but INTERNAL phase has ended.
        (0.009900, (0, 1, 0, 0)),
    ],
)
def test_4_sync(t_ext: float, expected: tuple[int, int, int, int]):
    """Test TimingModeSync in 4-node topology."""
    timing = TimingModeSync(t_ext=t_ext, t_int=0.010000 - t_ext)
    net, simulator = build_linear_network(4, t_cohere=0.015000, fw={"p_swap": 1.0}, timing=timing, end_time=0.029999)
    fwA, fwB, fwC, fwD = (node.get_app(ProactiveForwarder) for node in net.nodes)

    install_path(net, RoutingPathStatic("ABCD", swap=[2, 0, 1, 2]))
    provide_entanglements(
        ([0.001000] * 3, (fwA, fwB, fwC, fwD)),
    )
    simulator.run()
    print_fw_counters(net)
    check_fw_counters(
        net,
        n_consumed=(expected[0], 0, 0, expected[3]),
        n_swapped=(0, expected[1], expected[2], 0),
    )


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
    fwA, fwB, fwC, fwD = (node.get_app(ProactiveForwarder) for node in net.nodes)
    fwC.swap.ps = ps3

    install_path(net, RoutingPathStatic("ABCD"))
    provide_entanglements(
        (etg_ms, (fwA, fwB, fwC, fwD)),
        transform_t=lambda ms: 1 + ms / 1000,
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
            n_su_same=(0, 1, (0, 1), 0),  # if B hears C's failure before its own swap, it does not herald C
            n_su_lower=(1, 0, 0, 1),
        )
    check_memory_released(net)


@pytest.mark.parametrize(
    ("ps3", "delay3", "n_swap2", "n_consumed", "t_release", "n_cpacket"),
    [
        # 1. t=1.0110, B-C arrives, both B and C start swapping.
        # 2. t=1.0110, C completes swapping with success, heralds B for B-D.
        # 3. t=1.0115, B receives B-D heralding.
        # 4. t=1.0610, B completes swapping with success, heralds A for A-D, heralds C for A-C.
        # 5. t=1.0615, A receives A-D heralding, consumes EPR.
        # 6. t=1.0615, C receives A-C heralding, heralds D for A-D.
        # 7. t=1.0620, D receives A-D heralding, consumes EPR.
        (1.0, 0.0000, 1, 1, (1.0615, 1.0610, 1.0110, 1.0620), (1, 1, 1, 1)),
        # 1. t=1.0110, B-C arrives, both B and C start swapping.
        # 2. t=1.0110, C completes swapping with failure, heralds B+D for failure.
        # 3. t=1.0115, D receives failure heralding, releases qubit.
        # 4. t=1.0115, B receives failure heralding, heralds A for failure.
        # 5. t=1.0120, A receives failure heralding, releases qubit.
        # 4. t=1.0610, B completes swapping with success, ignores due to earlier failure.
        (0.0, 0.0000, 1, 0, (1.0120, 1.0610, 1.0110, 1.0115), (1, 1, 0, 1)),
        # 1. t=1.0110, B-C arrives, both B and C start swapping.
        # 2. t=1.0609, C completes swapping with success, heralds B for B-D.
        # 3. t=1.0610, B completes swapping with success, heralds C for A-C.
        # 4. t=1.0614, B receives B-D heralding, heralds A for A-D.
        # 5. t=1.0615, C receives A-C heralding, heralds D for A-D.
        # 6. t=1.0619, A receives A-D heralding, consumes EPR.
        # 7. t=1.0620, D receives A-D heralding, consumes EPR.
        (1.0, 0.0499, 1, 1, (1.0619, 1.0610, 1.0609, 1.0620), (1, 1, 1, 1)),
        # 1. t=1.0110, B-C arrives, both B and C start swapping.
        # 2. t=1.0609, C completes swapping with failure, heralds B+D for failure.
        # 3. t=1.0610, B completes swapping with success, heralds C for A-C.
        # 4. t=1.0614, D receives failure heralding, releases qubit.
        # 5. t=1.0614, B receives failure heralding, heralds A for failure.
        # 6. t=1.0615, C receives A-C heralding, ignores due to earlier failure.
        # 6. t=1.0619, A receives failure heralding, releases EPR.
        (0.0, 0.0499, 1, 0, (1.0619, 1.0610, 1.0609, 1.0614), (1, 1, 1, 1)),
        # 1. t=1.0110, B-C arrives, both B and C start swapping.
        # 2. t=1.0610, B completes swapping with success, heralds C for A-C.
        # 3. t=1.0611, C completes swapping with success, heralds B for B-D.
        # 4. t=1.0615, C receives A-C heralding, heralds D for A-D.
        # 5. t=1.0616, B receives B-D heralding, heralds A for A-D.
        # 6. t=1.0620, D receives A-D heralding, consumes EPR.
        # 7. t=1.0621, A receives A-D heralding, consumes EPR.
        (1.0, 0.0501, 1, 1, (1.0621, 1.0610, 1.0611, 1.0620), (1, 1, 1, 1)),
        # 1. t=1.0110, B-C arrives, both B and C start swapping.
        # 2. t=1.0610, B completes swapping with success, heralds C for A-C.
        # 3. t=1.0611, C completes swapping with failure, heralds B+D for failure.
        # 4. t=1.0615, C receives A-C heralding, ignores due to earlier failure.
        # 5. t=1.0616, D receives failure heralding, releases qubit.
        # 6. t=1.0616, B receives failure heralding, heralds A for failure.
        # 7. t=1.0621, A receives failure heralding, releases qubit.
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
    fwA, fwB, fwC, fwD = (node.get_app(ProactiveForwarder) for node in net.nodes)
    fwC.swap.ps = ps3
    fwC.swap.delay = ConstantDelayModel(delay3)
    fwC.swap.error = PerfectErrorModel()

    fwB_n_swapped_values: list[int] = []

    def save_counter():
        fwB_n_swapped_values.append(fwB.cnt.n_swapped)

    timer = Timer("save_counters", start_time=1.018, end_time=1.088, step_time=0.010, trigger_func=save_counter)
    timer.install(simulator)

    install_path(net, RoutingPathStatic("ABCD"))
    provide_entanglements(
        (1.000, fwA, fwB),
        (1.000, fwC, fwD),
        (1.010, fwB, fwC),
        fidelity=1,
    )
    cpacket_cnt = collect_cpacket_counts(monkeypatch)
    simulator.run()
    print_fw_counters(net)
    check_memory_released(net)
    print("cpacket_cnt", cpacket_cnt)

    assert fwB_n_swapped_values == [0, 0, 0, 0, 0, n_swap2, n_swap2, n_swap2]
    assert fwA.cnt.n_consumed == n_consumed
    if n_consumed > 0:
        assert 0.5 < fwA.cnt.consumed_avg_fidelity <= 0.75
        assert fwA.cnt.consumed_avg_fidelity == pytest.approx(fwD.cnt.consumed_avg_fidelity)

    assert list(fw.node.get_app(QubitReleaseReset).last_t for fw in (fwA, fwB, fwC, fwD)) == [
        simulator.time(sec=t) for t in t_release
    ]
    assert (cpacket_cnt["*-A"], cpacket_cnt["*-B"], cpacket_cnt["*-C"], cpacket_cnt["*-D"]) == n_cpacket


@pytest.mark.parametrize(
    ("swap_delay", "n_consumed"),
    [
        # 1. t=1.0010, elementary EPRs arrive, B starts swapping.
        # 2. t=1.0012, B/C completes swapping, heralds success to C/B.
        # 3. t=1.0017, C/B receives heralding, heralds success to D/A.
        # 4. t=1.0022, D/A receives heralding, consumes EPR.
        # 5. t=1.0030, memory qubits would decohere, but it's already consumed.
        (0.0002, 1),
        # 1. t=1.0010, elementary EPRs arrive, B starts swapping.
        # 2. t=1.0022, B/C completes swapping, heralds success to C/B.
        # 3. t=1.0027, C/B receives heralding, heralds success to D/A.
        # 4. t=1.0030, memory qubits decohere.
        # 5. t=1.0032, D/A receives heralding, ignores due to qubit not in memory.
        (0.0012, 0),
        # 1. t=1.0010, elementary EPRs arrive, B starts swapping.
        # 2. t=1.0027, B/C completes swapping, heralds success to C/B.
        # 3. t=1.0030, memory qubits decohere.
        # 4. t=1.0032, C/B receives heralding, heralds success to D/A.
        # 5. t=1.0037, D/A receives heralding, ignores due to qubit not in memory.
        (0.0017, 0),
        # 1. t=1.0010, elementary EPRs arrive, B starts swapping.
        # 2. t=1.0030, memory qubits decohere.
        # 3. t=1.0032, B/C aborts swapping, heralds failure to C+A/B+D.
        # 4. t=1.0037, C/B receives heralding, ignores due to earlier failure.
        # 5. t=1.0037, A/D receives heralding, ignores due to qubit not in memory.
        (0.0022, 0),
    ],
)
def test_4_decohere(swap_delay: float, n_consumed: int):
    """Test short decoherence time in 4-node topology."""
    net, simulator = build_linear_network(4, t_cohere=0.003, fw={"p_swap": 1.0, "swap_delay": swap_delay}, end_time=2)
    fwA, fwB, fwC, fwD = (node.get_app(ProactiveForwarder) for node in net.nodes)

    install_path(net, RoutingPathStatic("ABCD"))
    provide_entanglements(
        ([1.000] * 3, (fwA, fwB, fwC, fwD)),
    )
    simulator.run()
    print_fw_counters(net)
    check_fw_counters(
        net,
        n_consumed=(n_consumed, 0, 0, n_consumed),
        n_su_lower=(1, 0, 0, 1),
    )
    check_memory_released(net)


@pytest.mark.parametrize(("ps3", "etg_ms"), itertools.product((1, 0), ((2, 1, 1, 2), (1, 2, 2, 1))))
def test_5_asap(
    ps3: float,
    etg_ms: tuple[int, int, int, int],
):
    """Test SWAP-ASAP in 5-node topology with various entanglement arrival orders."""
    n_consumed = 1 if ps3 == 1 else 0

    net, simulator = build_linear_network(5, fw={"p_swap": 1.0}, end_time=2)
    fwA, fwB, fwC, fwD, fwE = (node.get_app(ProactiveForwarder) for node in net.nodes)
    fwC.swap.ps = ps3

    install_path(net, RoutingPathStatic("ABCDE"))
    provide_entanglements(
        (etg_ms, (fwA, fwB, fwC, fwD, fwE)),
        transform_t=lambda ms: 1 + ms / 1000,
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
    fwA, fwB, fwC, fwD, fwE = (node.get_app(ProactiveForwarder) for node in net.nodes)

    install_path(net, RoutingPathStatic("ABCDE", swap=swap))
    provide_entanglements(
        (etg_ms, (fwA, fwB, fwC, fwD, fwE)),
        transform_t=lambda ms: 1 + ms / 1000,
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
    ("etg_ms", "swap_delay", "cutoff4", "t_release", "n_consumed"),
    [
        # 1. t=1.0010, A-B and B-C EPRs arrive, B starts swapping.
        # 2. t=1.0015, B completes swapping, heralds success to C.
        #              B saves SwapTask awaiting heralding from C.
        # 3. t=1.0020, C receives heralding, cannot swap due to lack of C-D EPR.
        # 4. t=1.0100, memory qubits decohere at A and C.
        # 5. t=1.0200, B deletes SwapTask.
        ((0, 0, -1, -1), 0.0005, None, (1.0100, 1.0015, 1.0100), 0),
        # 1. t=1.0010, A-B and B-C EPRs arrive, B starts swapping.
        # 2. t=1.0097, B completes swapping, heralds success to C.
        #              B saves SwapTask awaiting heralding from C.
        # 3. t=1.0100, memory qubits decohere at A and C.
        # 4. t=1.0102, C receives heralding, ignores due to qubit not in memory.
        # 5. t=1.0200, B deletes SwapTask.
        ((0, 0, -1, -1), 0.0087, None, (1.0100, 1.0097, 1.0100), 0),
        # 1. t=1.0010, A-B and B-C and C-D EPRs arrive, B and C start swapping.
        # 2. t=1.0012, C completes swapping, cannot herald either side.
        #              C saves SwapTask awaiting heralding from B and D.
        # 3. t=1.0012, B completes swapping, heralds success to C.
        #              B saves SwapTask awaiting heralding from C.
        # 4. t=1.0017, C receives B heralding, heralds success to D.
        # 5. t=1.0022, D receives C heralding, cannot swap due to lack of D-E EPR.
        # 6. t=1.0100, memory qubits decohere at A and D.
        ((0, 0, 0, -1), 0.0002, None, (1.0100, 1.0012, 1.0012, 1.0100), 0),
        # 1. t=1.0010, A-B and B-C and C-D EPRs arrive, B and C start swapping.
        # 2. t=1.0012, C completes swapping, cannot herald either side.
        #              C saves SwapTask awaiting heralding from B and D.
        # 3. t=1.0012, B completes swapping, heralds success to C.
        #              B saves SwapTask awaiting heralding from C.
        # 4. t=1.0017, C receives B heralding, heralds success to D.
        # 5. t=1.0020, D discards qubit due to exceeding wait-time cut-off.
        # 6. t=1.0022, D receives C heralding, cannot swap due to lack of D-E EPR.
        # 7. t=1.0100, memory qubit decoheres at A.
        #              XXX Ideally, A should be informed so that it can release its qubit earlier.
        ((0, 0, 0, -1), 0.0002, 0.0010, (1.0100, 1.0012, 1.0012, 1.0020), 0),
    ],
)
def test_5_decohere(
    etg_ms: tuple[int, int, int], swap_delay: float, cutoff4: float | None, t_release: tuple[float, ...], n_consumed: int
):
    """Test short decoherence time in 5-node topology."""
    net, simulator = build_linear_network(5, t_cohere=0.010, fw={"p_swap": 1.0, "swap_delay": swap_delay}, end_time=2)
    fwA, fwB, fwC, fwD, fwE = (node.get_app(ProactiveForwarder) for node in net.nodes)

    swap_cutoff = None if cutoff4 is None else [-1, -1, -1, cutoff4, -1]
    install_path(net, RoutingPathStatic("ABCDE", swap_cutoff=swap_cutoff))
    provide_entanglements(
        (etg_ms, (fwA, fwB, fwC, fwD, fwE)),
        transform_t=lambda ms: 1 + ms / 1000,
    )
    simulator.run()
    print_fw_counters(net)
    check_fw_counters(
        net,
        n_consumed=(n_consumed, 0, 0, 0, n_consumed),
        # n_su_lower=(1, 0, 0, 0, 1),
    )
    check_memory_released(net)
    assert list(node.get_app(QubitReleaseReset).last_t for node in net.nodes[: len(t_release)]) == [
        simulator.time(sec=t) for t in t_release
    ]


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
    fwA, fwB, fwC, fwD = (node.get_app(ProactiveForwarder) for node in net.nodes)

    rp = install_path(net, RoutingPathMulti("A", "D"))

    def check_fib_entries():
        routes = {"-".join(fwA.fib.get(path_id).route) for path_id in (rp.path_id, rp.path_id + 1)}
        assert routes == {"A-B-D", "A-C-D"}

    simulator.add_event(func_to_event(simulator.time(sec=2.0), check_fib_entries))
    provide_entanglements(
        (1.001 if has_etg[0] else -1, fwA, fwB),
        (1.002 if has_etg[1] else -1, fwB, fwD),
        (1.001 if has_etg[2] else -1, fwA, fwC),
        (1.002 if has_etg[3] else -1, fwC, fwD),
    )
    simulator.run()
    print_fw_counters(net)

    assert fwA.cnt.n_consumed == n_consumed == fwD.cnt.n_consumed
    assert (fwB.cnt.n_swapped, fwC.cnt.n_swapped) == n_swapped


@pytest.mark.parametrize(
    ("t_edge_etg", "selected_path", "n_consumed"),
    [
        # Both B-A and A-C channels select the same path,
        # so that end-to-end entanglements are created on the selected path,
        # regardless of whether edge entanglements arrive before or after center entanglements.
        (1.001, (0, 0), (1, 0)),
        (1.001, (1, 1), (0, 1)),
        (1.007, (0, 0), (1, 0)),
        (1.007, (1, 1), (0, 1)),
        # The B-A and A-C channels select different paths,
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
        elif epr.src is fwB.node:  # B-A
            chosen = (rp0.path_id, rp1.path_id)[selected_path[0]]
        else:  # A-C
            assert epr.src is fwA.node
            chosen = (rp0.path_id, rp1.path_id)[selected_path[1]]
        return chosen

    net, simulator = build_tree_network(
        fw={"p_swap": 1.0, "mux": MuxSchemeDynamicEpr(select_path=select_path)},
        # If there is a conflict in path selection, there will be two leftover EPRs.
        swap_table_leak_tol=2 if selected_path[0] != selected_path[1] else 0,
    )
    fwA, fwB, fwC, fwD, fwE, fwF, fwG = (node.get_app(ProactiveForwarder) for node in net.nodes)

    rp0 = install_path(net, RoutingPathStatic("DBACF", m_v=QubitAllocationType.DISABLED))
    rp1 = install_path(net, RoutingPathStatic("EBACG", m_v=QubitAllocationType.DISABLED))

    provide_entanglements(
        (t_edge_etg, fwD, fwB),
        (t_edge_etg, fwE, fwB),
        (t_edge_etg, fwC, fwF),
        (t_edge_etg, fwC, fwG),
        (1.003, fwB, fwA),
        (1.005, fwA, fwC),
    )
    simulator.run()
    print_fw_counters(net)

    assert fwD.cnt.n_consumed == n_consumed[0] == fwF.cnt.n_consumed
    assert fwE.cnt.n_consumed == n_consumed[1] == fwG.cnt.n_consumed


@pytest.mark.parametrize(
    ("etg_ms", "selected_qubit", "selected_path", "n_consumed"),
    [
        # EPRs arrive in left-to-right order: D-B & E-B, B-A, A-C, C-F & C-G.
        # C, A, B perform their swaps sequentially.
        ((1, 1, 2, 3, 4, 5), (0, 9), -1, (1, 0)),
        ((1, 1, 2, 3, 5, 4), (0, 9), -1, (1, 0)),
        ((1, 1, 2, 3, 4, 5), (1, 9), -1, (0, 1)),
        ((1, 1, 2, 3, 5, 4), (1, 9), -1, (0, 1)),
        # EPRs arrive in outer-to-inner order: D-B / E-B, C-F / C-G, B-A & A-C.
        # B and C swap first in parallel (without choice), and A swaps last.
        ((1, -1, 2, 2, 1, -1), (9, 9), -1, (1, 0)),
        ((-1, 1, 2, 2, -1, 1), (9, 9), -1, (0, 1)),
        ((1, -1, 2, 2, -1, 1), (9, 9), -1, (0, 0)),
        # EPRs arrive in inner-to-outer order: B-A & A-C, D-B / E-B, C-F / C-G.
        # A swaps first, B and C swap last in parallel.
        ((2, -1, 1, 1, 2, -1), (9, 9), -1, (1, 0)),
        ((-1, 2, 1, 1, -1, 2), (9, 9), -1, (0, 1)),
        ((2, -1, 1, 1, -1, 2), (9, 9), -1, (0, 0)),
        # EPRs arrive in inner-to-outer order: B-A & A-C, D-B & E-B, C-F & C-G.
        # A swaps first, B and C swap last in parallel but with physically unrealistic coordinated decisions.
        ((2, 3, 1, 1, 3, 2), (9, 9), 0, (1, 0)),
        ((2, 3, 1, 1, 3, 2), (9, 9), 1, (0, 1)),
    ],
)
def test_tree2_statistical(
    etg_ms: tuple[int, int, int, int, int, int],
    selected_qubit: tuple[int, int],
    selected_path: int,
    n_consumed: tuple[int, int],
):
    """
    Test MuxSchemeStatistical in tree (height=2) topology with two 4-link paths.
    """

    def select_qubit(candidates: list[MemoryEprTuple], fw: Forwarder, mt0: MemoryEprTuple) -> MemoryEprTuple:
        _ = mt0
        assert len(candidates) == 2
        if fw is fwB:  # B-A choosing between D-B and E-B
            partner = (fwD, fwE)[selected_qubit[0]]
            chosen = next((mt1 for mt1 in candidates if mt1[1].src is partner.node), None)
        elif fw is fwC:  # A-C choosing between C-F and C-G
            partner = (fwF, fwG)[selected_qubit[1]]
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
    fwA, fwB, fwC, fwD, fwE, fwF, fwG = (node.get_app(ProactiveForwarder) for node in net.nodes)

    install_path(net, RoutingPathStatic("DBACF", m_v=QubitAllocationType.DISABLED))
    install_path(net, RoutingPathStatic("EBACG", m_v=QubitAllocationType.DISABLED))

    edges = ((fwD, fwB), (fwE, fwB), (fwB, fwA), (fwA, fwC), (fwC, fwF), (fwC, fwG))
    provide_entanglements(
        *((t, *edge) for (t, edge) in zip(etg_ms, edges)),
        transform_t=lambda ms: 1 + ms / 1000,
    )
    simulator.run()
    print_fw_counters(net)
    if -1 in etg_ms:
        # For test cases without unused EPRs, swap conflict should release memory quickly.
        check_memory_released(net)

    assert fwD.cnt.n_consumed == n_consumed[0] == fwF.cnt.n_consumed
    assert fwE.cnt.n_consumed == n_consumed[1] == fwG.cnt.n_consumed
    assert sum(fw.cnt.n_su_lower[4] for fw in (fwD, fwE, fwF, fwG)) == (2 if sum(n_consumed) == 0 else 0)


@pytest.mark.parametrize(
    ("path0", "path1", "etgs", "selected_qubit", "selected_path", "n_consumed"),
    [
        # H--\     /--A
        #     D---B
        # I--/     \--E---K
        #
        # 1. D chooses H-D over I-D, swaps H-D-B, heralds B for H-B q_paths={0}.
        # 2. B ignores B-E, swaps H-B-A, heralds H (via D) and A for H-A q_paths={0}.
        ("HDBA", "IDBEK", "HD,ID:DB:BE:BA", {"D": "H"}, {}, (1, 0)),
        # 1. D chooses I-D over H-D, swaps I-D-B, heralds B for I-B q_paths={1}.
        # 2. B ignores B-A, swaps I-B-E, heralds E for I-E q_paths={1}.
        # 3. E swaps I-E-K, heralds I (via B,D) and K for I-K q_paths={1}.
        ("HDBA", "IDBEK", "HD,ID:DB:BA,BE:EK", {"D": "I"}, {}, (0, 1)),
        #
        # H---D--\         /--F
        #         B---A---C
        #     E--/         \--G--N
        #
        # 1. Outer EPRs arrive first.
        #    D swaps H-D-B and heralds B for H-B q_paths={0}.
        #    G swaps C-G-N and heralds C for C-N q_paths={1}.
        # 2. B,A,C swaps at the same time.
        #    B chooses H-D-B over E-B, heralds A for H-A q_paths={0}.
        #    C chooses C-F over C-G-N, heralds A for A-F q_paths={0}.
        #    A chooses path0 as representative.
        # 3. A receives heralding from C, heralds H (via B,D) for H-F q_paths={0}.
        #    A receives heralding from B, heralds F (via C) for H-F q_paths={0}.
        ("HDBACF", "EBACGN", "HD,DB,EB,CF,CG,GN:BA,AC", {"B": "D", "C": "F"}, {"A": 0}, (1, 0)),
        # Similar, but A chooses path1 as representative.
        ("HDBACF", "EBACGN", "HD,DB,EB,CF,CG,GN:BA,AC", {"B": "D", "C": "F"}, {"A": 1}, (1, 0)),
        # Similar, but B and C choose path1 qubits.
        ("HDBACF", "EBACGN", "HD,DB,EB,CF,CG,GN:BA,AC", {"B": "E", "C": "G"}, {"A": 0}, (0, 1)),
        # Similar, but B and C make conflicting choices.
        ("HDBACF", "EBACGN", "HD,DB,EB,CF,CG,GN:BA,AC", {"B": "D", "C": "G"}, {"A": 0}, (0, 0)),
    ],
)
def test_tree3_statistical(
    path0: str,
    path1: str,
    etgs: str,
    selected_qubit: dict[str, str],
    selected_path: dict[str, int],
    n_consumed: tuple[int, int],
):
    """
    Test MuxSchemeStatistical in tree (height=3) topology with uneven paths.
    """

    def select_qubit(candidates: list[MemoryEprTuple], fw: Forwarder, mt0: MemoryEprTuple) -> MemoryEprTuple:
        _ = mt0
        partner = selected_qubit[fw.node.name]
        chosen = next((mt1 for mt1 in candidates if partner in (cast(QNode, mt1[1].src).name, cast(QNode, mt1[1].dst).name)))
        assert chosen is not None
        return chosen

    def select_path(candidates: list[int], fw: Forwarder, epr0: Entanglement, epr1: Entanglement) -> int:
        _ = candidates, epr0, epr1
        return selected_path[fw.node.name]

    net, simulator = build_tree_network(
        height=3,
        fw={"p_swap": 1.0, "mux": MuxSchemeStatistical(select_swap_qubit=select_qubit, select_path=select_path)},
        end_time=2,
        swap_table_leak_tol=2,
    )
    fws = {node.name: node.get_app(ProactiveForwarder) for node in net.nodes}

    for path in path0, path1:
        install_path(net, RoutingPathStatic(path, m_v=QubitAllocationType.DISABLED))

    def expand_etgs():
        for t, edges in enumerate(etgs.split(":")):
            for edge in edges.split(","):
                yield t, fws[edge[0]], fws[edge[1]]

    provide_entanglements(*expand_etgs(), transform_t=lambda ms: 1 + ms / 1000)
    simulator.run()
    print_fw_counters(net)

    assert fws[path0[0]].cnt.n_consumed == n_consumed[0]
    assert fws[path1[0]].cnt.n_consumed == n_consumed[1]
