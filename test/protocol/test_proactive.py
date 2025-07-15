import pytest

from qns.entity import Controller
from qns.network.network import ClassicTopology, QuantumNetwork
from qns.network.protocol.link_layer import LinkLayer
from qns.network.protocol.proactive_forwarder import ProactiveForwarder
from qns.network.protocol.proactive_routing_controller import ProactiveRoutingControllerApp
from qns.network.topology import LinearTopology
from qns.simulator import Simulator
from qns.utils import log


def build_linear_network(
    n_nodes: int,
    *,
    qchannel_capacity=1,
) -> tuple[QuantumNetwork, Simulator]:
    topo = LinearTopology(
        nodes_number=n_nodes,
        nodes_apps=[LinkLayer(), ProactiveForwarder(ps=0.5)],
        qchannel_args={"length": 100},  # delay is 0.0005 seconds
        cchannel_args={"length": 100},
        memory_args={"decoherence_rate": 1 / 5.0, "capacity": 2 * qchannel_capacity},
    )
    topo.controller = Controller("ctrl", apps=[ProactiveRoutingControllerApp(swapping=[])])

    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)
    for qchannel in net.get_qchannels():
        qchannel.assign_memory_qubits(capacity=qchannel_capacity)
    topo.connect_controller(net.get_nodes())

    simulator = Simulator(0.0, 60.0)
    log.install(simulator)
    net.install(simulator)

    return net, simulator


def test_proactive_path_validation():
    """Test controller path validation logic."""
    net, _ = build_linear_network(5)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)

    with pytest.raises(ValueError, match="swapping order"):
        ctrl.install_path_on_route([], path_id=0, swap=[])

    with pytest.raises(ValueError, match="swapping order"):
        ctrl.install_path_on_route(["n1", "n2", "n3", "n4", "n5"], path_id=0, swap=[0, 0, 0])

    with pytest.raises(ValueError, match="purif segment r1-r2"):
        ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, swap=[1, 0, 1], purif={"r1-r2": 1})

    with pytest.raises(ValueError, match="purif segment n1-n2-n3"):
        ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, swap=[1, 0, 1], purif={"n1-n2-n3": 1})

    with pytest.raises(ValueError, match="purif segment n2-n2"):
        ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, swap=[1, 0, 1], purif={"n2-n2": 1})

    with pytest.raises(ValueError, match="purif segment n3-n1"):
        ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, swap=[1, 0, 1], purif={"n3-n1": 1})


def test_proactive_isolated():
    """Test isolated links mode where swapping is disabled."""
    net, simulator = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, swap=[0, 0, 0])
    simulator.run()

    for app in (f1, f2, f3):
        print((app.own.name, app.cnt))

    assert f1.cnt.n_entg == f1.cnt.n_eligible == f1.cnt.n_consumed > 0
    assert f2.cnt.n_entg == f2.cnt.n_eligible == f2.cnt.n_consumed > 0
    assert f3.cnt.n_entg == f3.cnt.n_eligible == f3.cnt.n_consumed > 0
    assert f1.cnt.n_swapped == f2.cnt.n_swapped == f3.cnt.n_swapped == 0
    assert f1.cnt.n_consumed + f3.cnt.n_consumed == f2.cnt.n_consumed


def test_proactive_basic():
    """Test basic swapping."""
    net, simulator = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, swap=[1, 0, 1])
    simulator.run()

    for app in (f1, f2, f3):
        print((app.own.name, app.cnt))

    # entanglements at n2 are immediately eligible because there's no purification
    assert f1.cnt.n_entg + f3.cnt.n_entg == f2.cnt.n_entg == f2.cnt.n_eligible
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.8
    # no swapping is expected at n1 and n3
    assert f1.cnt.n_swapped == f3.cnt.n_swapped == 0
    # successful swap at n2 should make the qubit eligible at n1 and n3, but allow 1 lost at end of simulation
    assert 0 <= f2.cnt.n_swapped - f1.cnt.n_eligible <= 1
    # eligible qubits at n1 and n3 are immediately consumed
    assert f1.cnt.n_eligible == f3.cnt.n_eligible == f1.cnt.n_consumed == f3.cnt.n_consumed >= 15
    # consumed entanglement should have expected fidelity above 0.7
    assert 0.7 <= f1.cnt.consumed_avg_fidelity == pytest.approx(f3.cnt.consumed_avg_fidelity, abs=1e-3)
    # no consumption is expected at n2
    assert f2.cnt.n_consumed == 0


def test_proactive_parallel():
    """Test parallel swapping."""
    net, simulator = build_linear_network(4)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    ctrl.install_path_on_route(["n1", "n2", "n3", "n4"], path_id=0, swap=[1, 0, 0, 1])
    simulator.run()

    for app in (f1, f2, f3, f4):
        print((app.own.name, app.cnt))

    # entanglements at n2 and n3 are immediately eligible because there's no purification
    assert f2.cnt.n_entg == f2.cnt.n_eligible
    assert f3.cnt.n_entg == f3.cnt.n_eligible
    # only eligible qubits may be swapped at n2 and n3, with 50% success rate
    assert f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.8
    assert f3.cnt.n_swapped <= f3.cnt.n_eligible * 0.8
    # some swaps were completed in parallel
    assert f2.cnt.n_swapped_p > 0
    assert f3.cnt.n_swapped_p > 0
    # successful swap at both n2 and n3 can make the qubit eligible at n1 and n4,
    # but there would be losses because parallel swaps require coincidence
    assert min(f2.cnt.n_swapped, f3.cnt.n_swapped) > f1.cnt.n_eligible > 0
    # eligible qubits at n1 and n4 are immediately consumed
    assert f1.cnt.n_eligible == f4.cnt.n_eligible == f1.cnt.n_consumed == f4.cnt.n_consumed >= 10
    # consumed entanglement should have expected fidelity above 0.6
    assert 0.6 <= f1.cnt.consumed_avg_fidelity == pytest.approx(f4.cnt.consumed_avg_fidelity, abs=1e-3)


def test_proactive_purif_link1r():
    """Test 1-round purification on each link."""
    net, simulator = build_linear_network(3, qchannel_capacity=2)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, swap=[1, 0, 1], purif={"n1-n2": 1, "n2-n3": 1})
    simulator.run()

    for app in (f1, f2, f3):
        print((app.own.name, app.cnt))

        # some purifications should fail
        assert app.cnt.n_purif[0] < app.cnt.n_entg * 0.8

    # entanglements at n2 are eligible after 1-round purification
    assert pytest.approx(f1.cnt.n_purif[0] + f3.cnt.n_purif[0], abs=1) == f2.cnt.n_purif[0] == f2.cnt.n_eligible
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.8
    # successful swap at n2 should make the qubit eligible at n1 and n3, but allow 1 lost at end of simulation
    assert 0 <= f2.cnt.n_swapped - f1.cnt.n_eligible <= 1
    # eligible qubits at n1 and n3 are immediately consumed
    assert f1.cnt.n_eligible == f3.cnt.n_eligible == f1.cnt.n_consumed == f3.cnt.n_consumed >= 15
    # consumed entanglement should have expected fidelity above 0.7
    assert 0.7 <= f1.cnt.consumed_avg_fidelity == pytest.approx(f3.cnt.consumed_avg_fidelity, abs=1e-3)


def test_proactive_purif_link2r():
    """Test 2-round purification on each link."""
    net, simulator = build_linear_network(3, qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, swap=[1, 0, 1], purif={"n1-n2": 2, "n2-n3": 2})
    simulator.run()

    for app in (f1, f2, f3):
        print((app.own.name, app.cnt))

        # some purifications should fail
        assert app.cnt.n_purif[0] < app.cnt.n_entg * 0.8
        assert app.cnt.n_purif[1] < app.cnt.n_purif[0] * 0.8

    # entanglements at n2 are eligible after 2-round purification
    assert pytest.approx(f1.cnt.n_purif[0] + f3.cnt.n_purif[0], abs=1) == f2.cnt.n_purif[0]
    assert pytest.approx(f1.cnt.n_purif[1] + f3.cnt.n_purif[1], abs=1) == f2.cnt.n_purif[1] == f2.cnt.n_eligible
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.8
    # successful swap at n2 should make the qubit eligible at n1 and n3, but allow 1 lost at end of simulation
    assert 0 <= f2.cnt.n_swapped - f1.cnt.n_eligible <= 1
    # eligible qubits at n1 and n3 are immediately consumed
    assert f1.cnt.n_eligible == f3.cnt.n_eligible == f1.cnt.n_consumed == f3.cnt.n_consumed >= 15
    # consumed entanglement should have expected fidelity above 0.8
    assert 0.8 <= f1.cnt.consumed_avg_fidelity == pytest.approx(f3.cnt.consumed_avg_fidelity, abs=1e-3)


def test_proactive_purif_ee2r():
    """Test 2-round purification between two end nodes."""
    net, simulator = build_linear_network(3, qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, swap=[1, 0, 1], purif={"n1-n3": 2})
    simulator.run()

    for app in (f1, f2, f3):
        print((app.own.name, app.cnt))

    # successful swap at n2 enables first round of n1-n3 purification; some purifications should fail
    assert pytest.approx(f1.cnt.n_purif[0], abs=1) == f3.cnt.n_purif[0] < f2.cnt.n_swapped * 0.8
    assert pytest.approx(f1.cnt.n_purif[1], abs=1) == f3.cnt.n_purif[1] < f3.cnt.n_purif[0] * 0.8
    # no purification may occur in n2
    assert len(f2.cnt.n_purif) == 0
    # eligible qubits at n1 and n3 are consumed after completing 2-round purification
    assert f1.cnt.n_eligible == f3.cnt.n_eligible == f1.cnt.n_consumed == f3.cnt.n_consumed >= 5
    # consumed entanglement should have expected fidelity above 0.8
    assert 0.8 <= f1.cnt.consumed_avg_fidelity == pytest.approx(f3.cnt.consumed_avg_fidelity, abs=1e-3)
