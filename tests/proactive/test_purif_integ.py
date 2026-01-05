"""
Test suite for purification in proactive forwarding, integrated with LinkLayer.
"""

import pytest

from mqns.network.proactive import ProactiveForwarder, RoutingPathStatic

from .proactive_common import build_linear_network, check_e2e_consumed, install_path


def test_purif_link1r():
    """Test 1-round purification on each link."""
    net, simulator = build_linear_network(3, qchannel_capacity=2)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(net, RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], purif={"n1-n2": 1, "n2-n3": 1}))
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)

        # some purifications should fail
        assert fw.cnt.n_purif[0] < fw.cnt.n_entg * 0.8

    # entanglements at n2 are eligible after 1-round purification
    f1f3_n_purif0 = f1.cnt.n_purif[0] + f3.cnt.n_purif[0]
    assert f1f3_n_purif0 >= f2.cnt.n_purif[0] == f2.cnt.n_eligible >= f1f3_n_purif0 - 2
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert f2.cnt.n_eligible * 0.7 >= f2.cnt.n_swapped >= 1000
    # successful swap at n2 should make the qubit eligible at n1 and n3;
    # eligible qubits at n1 and n3 are immediately consumed;
    # consumed entanglements should have expected fidelity better than swap_1
    check_e2e_consumed(f1, f3, n_swaps=f2.cnt.n_swapped, swap_balanced=True, has_purif=True, capacity=2, f_min=0.82, f_max=0.88)


def test_purif_link2r():
    """Test 2-round purification on each link."""
    net, simulator = build_linear_network(3, qchannel_capacity=4)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(net, RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], purif={"n1-n2": 2, "n2-n3": 2}))
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)

        # some purifications should fail
        assert fw.cnt.n_purif[0] < fw.cnt.n_entg * 0.8
        assert fw.cnt.n_purif[1] < fw.cnt.n_purif[0] * 0.8

    # entanglements at n2 are eligible after 2-round purification
    f1f3_n_purif0 = f1.cnt.n_purif[0] + f3.cnt.n_purif[0]
    assert f1f3_n_purif0 >= f2.cnt.n_purif[0] >= f1f3_n_purif0 - 4
    f1f3_n_purif1 = f1.cnt.n_purif[1] + f3.cnt.n_purif[1]
    assert f1f3_n_purif1 >= f2.cnt.n_purif[1] == f2.cnt.n_eligible >= f1f3_n_purif1 - 4
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert 0 < f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.7
    assert f2.cnt.n_swapped > 600
    # successful swap at n2 should make the qubit eligible at n1 and n3;
    # eligible qubits at n1 and n3 are immediately consumed;
    # consumed entanglements should have expected fidelity better than purif_link1r
    check_e2e_consumed(f1, f3, n_swaps=f2.cnt.n_swapped, swap_balanced=True, has_purif=True, capacity=4, f_min=0.86, f_max=0.92)


def test_purif_ee2r():
    """Test 2-round purification between two end nodes."""
    net, simulator = build_linear_network(3, qchannel_capacity=4)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(net, RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], purif={"n1-n3": 2}))
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)

    # successful swap at n2 enables first round of n1-n3 purification; some purifications should fail
    assert pytest.approx(f1.cnt.n_purif[0], abs=1) == f3.cnt.n_purif[0] < f2.cnt.n_swapped * 0.6
    assert pytest.approx(f1.cnt.n_purif[1], abs=1) == f3.cnt.n_purif[1] < f3.cnt.n_purif[0] * 0.6
    # no purification may occur in n2
    assert len(f2.cnt.n_purif) == 0
    # eligible qubits at n1 and n3 are consumed after completing 2-round purification;
    # consumed entanglements should have expected fidelity better than swap_1
    check_e2e_consumed(f1, f3, n_min=500, capacity=4, swap_balanced=True, has_purif=True, f_min=0.85, f_max=0.91)
