"""
Test suite for ProactiveForwarder integrated with LinkLayer.
"""

import itertools

import pytest

from mqns.entity.timer import Timer
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.network.network import TimingModeAsync, TimingModeSync
from mqns.network.proactive import ProactiveForwarder, RoutingPathSingle, RoutingPathStatic
from mqns.network.protocol.link_layer import LinkLayer

from .proactive_common import build_linear_network, build_rect_network, install_path, print_fw_counters


@pytest.mark.parametrize(
    ("epr_type", "timing_mode", "swap_order"),
    itertools.product(
        (WernerStateEntanglement, MixedStateEntanglement),
        ("ASYNC", "SYNC"),
        ("asap", "l2r", "r2l"),
    ),
)
def test_4_swap(epr_type: type[Entanglement], timing_mode: str, swap_order: str):
    """Test swapping in 4-node topology."""
    timing = TimingModeAsync() if timing_mode == "ASYNC" else TimingModeSync(t_ext=0.006, t_int=0.004)
    net, simulator = build_linear_network(4, end_time=3.0, timing=timing, epr_type=epr_type, has_link_layer=True)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    install_path(net, RoutingPathSingle("n1", "n4", swap=swap_order))
    simulator.run()
    print_fw_counters(net)

    # The main purpose of integrated test is to verify that the forwarder can return released qubits back to LinkLayer
    # for re-generating elementary entanglements.
    # Hence, these numeric bounds are much smaller than usual values, but must be greater than the memory capacity.
    assert f2.cnt.n_swapped >= 16
    assert f3.cnt.n_swapped >= 16
    assert f1.cnt.n_consumed >= 16
    assert f4.cnt.n_consumed >= 16
    assert -4 <= f1.cnt.n_consumed - f4.cnt.n_consumed <= 4


def test_rect_uninstall_path():
    """Test uninstall_path in rectangle topology."""
    net, simulator = build_rect_network(has_link_layer=True)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    ll1 = net.get_node("n1").get_app(LinkLayer)
    ll2 = net.get_node("n2").get_app(LinkLayer)
    ll3 = net.get_node("n3").get_app(LinkLayer)

    counters: list[tuple[int, int, int, int, int]] = []

    def save_counters():
        print_fw_counters(net)
        counters.append(
            (
                f2.cnt.n_swapped,
                ll2.cnt.n_attempts,
                f3.cnt.n_swapped,
                ll3.cnt.n_attempts,
                ll1.cnt.n_attempts,
            )
        )

    timer = Timer("save_counters", start_time=0.500, end_time=9.501, step_time=1.000, trigger_func=save_counters)
    timer.install(simulator)

    install_path(net, RoutingPathStatic(["n1", "n2", "n4"], swap=[1, 0, 1]), t_install=2, t_uninstall=6)
    install_path(net, RoutingPathStatic(["n1", "n3", "n4"], swap=[1, 0, 1]), t_install=4, t_uninstall=8)
    simulator.run()

    assert len(counters) == 10
    for i in 0, 1:  # f2.cnt.n_swapped and ll2.cnt.n_attempts
        assert counters[0][i] == counters[1][i]
        assert counters[6][i] == counters[9][i]
    for i in 2, 3:  # f3.cnt.n_swapped and ll3.cnt.n_attempts
        assert counters[0][i] == counters[3][i]
        assert counters[8][i] == counters[9][i]
    # ll1.cnt.n_attempts
    assert counters[0][4] == counters[1][4]
    assert counters[8][4] == counters[9][4]
