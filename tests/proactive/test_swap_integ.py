"""
Test suite for swapping in proactive forwarding, integrated with LinkLayer.
"""

from copy import deepcopy

import numpy as np

from mqns.entity.qchannel import LinkArchDimBk
from mqns.network.proactive import (
    CutoffSchemeWaitTime,
    ProactiveForwarder,
    ProactiveRoutingController,
    RoutingPathSingle,
)

from .proactive_common import (
    build_linear_network,
    dflt_qchannel_args,
    install_path,
)


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
