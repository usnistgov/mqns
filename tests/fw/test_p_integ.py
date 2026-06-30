"""
Test suite for ProactiveForwarder integrated with LinkLayer.
"""

import itertools

import pytest

from mqns.entity.timer import Timer
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.network.fw import ForwarderConsumeCounters, RoutingPathSingle, RoutingPathStatic, SwapSequenceInput
from mqns.network.network import TimingModeAsync, TimingModeSync
from mqns.network.proactive import ProactiveForwarder
from mqns.network.protocol.link_layer import LinkLayer

from .fw_common import build_linear_network, build_rect_network, install_path, print_fw_counters


@pytest.mark.parametrize(
    ("epr_type", "timing_mode", "swap"),
    list(
        itertools.product(
            (WernerStateEntanglement, MixedStateEntanglement),
            ("ASYNC", "SYNC"),
            ("asap", "l2r", "r2l"),
        )
    ),
)
def test_4_swap(epr_type: type[Entanglement], timing_mode: str, swap: SwapSequenceInput):
    """Test swapping in 4-node topology."""
    timing = TimingModeAsync() if timing_mode == "ASYNC" else TimingModeSync(t_ext=0.006, t_int=0.004)
    net, simulator = build_linear_network(
        4, swap_table_leak_tol=256, end_time=3.0, timing=timing, epr_type=epr_type, has_link_layer=True
    )
    _, fwB, fwC, _ = (node.get_app(ProactiveForwarder) for node in net.nodes)

    install_path(net, RoutingPathSingle("A", "D", swap=swap))
    simulator.run()
    print_fw_counters(net)

    # The main purpose of integrated test is to verify that the forwarder can return released qubits back to LinkLayer
    # for re-generating elementary entanglements.
    # Hence, these numeric bounds are much smaller than usual values, but must be greater than the memory capacity.
    assert fwB.cnt.n_swapped >= 16
    assert fwC.cnt.n_swapped >= 16
    consume_cnt = ForwarderConsumeCounters.of_path(net, "A", "D")
    assert consume_cnt.n_consumed >= 16


def test_rect_uninstall_path():
    """Test uninstall_path in rectangle topology."""
    net, simulator = build_rect_network(swap_table_leak_tol=256, has_link_layer=True)
    fwB = net.get_node("B").get_app(ProactiveForwarder)
    fwC = net.get_node("C").get_app(ProactiveForwarder)
    llA = net.get_node("A").get_app(LinkLayer)
    llB = net.get_node("B").get_app(LinkLayer)
    llC = net.get_node("C").get_app(LinkLayer)

    counters: list[tuple[int, int, int, int, int]] = []

    def save_counters():
        print_fw_counters(net)
        counters.append(
            (
                fwB.cnt.n_swapped,
                llB.cnt.n_attempts,
                fwC.cnt.n_swapped,
                llC.cnt.n_attempts,
                llA.cnt.n_attempts,
            )
        )

    timer = Timer("save_counters", start_time=0.500, end_time=9.501, step_time=1.000, trigger_func=save_counters)
    timer.install(simulator)

    install_path(net, RoutingPathStatic("ABD"), t_install=2, t_uninstall=6)
    install_path(net, RoutingPathStatic("ACD"), t_install=4, t_uninstall=8)
    simulator.run()

    assert len(counters) == 10
    for i in 0, 1:  # fwB.cnt.n_swapped and llB.cnt.n_attempts
        assert counters[0][i] == counters[1][i]
        assert counters[6][i] == counters[9][i]
    for i in 2, 3:  # fwC.cnt.n_swapped and llC.cnt.n_attempts
        assert counters[0][i] == counters[3][i]
        assert counters[8][i] == counters[9][i]
    # llA.cnt.n_attempts
    assert counters[0][4] == counters[1][4]
    assert counters[8][4] == counters[9][4]
