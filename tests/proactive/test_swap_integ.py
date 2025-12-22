"""
Test suite for swapping in proactive forwarding, integrated with LinkLayer.
"""

from copy import deepcopy

import numpy as np

from mqns.entity.qchannel import LinkArchDimBk
from mqns.network.proactive import (
    CutoffSchemeWaitTime,
    MuxSchemeDynamicEpr,
    ProactiveForwarder,
    ProactiveRoutingController,
    QubitAllocationType,
    RoutingPathMulti,
    RoutingPathSingle,
)

from .proactive_common import (
    CheckUnchanged,
    build_dumbbell_network,
    build_linear_network,
    build_rect_network,
    check_e2e_consumed,
    dflt_qchannel_args,
    install_path,
)


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
