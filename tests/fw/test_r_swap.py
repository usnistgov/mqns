"""
Test suite for ProactiveForwarder focused on swapping.
"""

from mqns.network.network import TimingModeSync
from mqns.network.reactive import ReactiveForwarder, ReactiveRoutingController

from .fw_common import (
    build_linear_network,
    print_fw_counters,
    provide_entanglements,
)


def test_3_minimal():
    """Test 3-node minimal swap."""
    net, simulator = build_linear_network(3, mode="R", ps=1.0, timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003))
    ctrl = net.get_controller().get_app(ReactiveRoutingController)
    f1 = net.get_node("n1").get_app(ReactiveForwarder)
    f2 = net.get_node("n2").get_app(ReactiveForwarder)
    f3 = net.get_node("n3").get_app(ReactiveForwarder)

    provide_entanglements(
        (1.001, f1, f2),
        (1.002, f2, f3),
    )
    simulator.run()
    print(ctrl.cnt)
    print_fw_counters(net)

    assert ctrl.cnt.n_ls == 3
    assert ctrl.cnt.n_decision == 1
    assert f1.cnt.n_consumed == 1
    assert f2.cnt.n_consumed == 0
    assert f3.cnt.n_consumed == 1
    assert f2.cnt.n_swapped == 1
