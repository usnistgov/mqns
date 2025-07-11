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
        ctrl.install_path_on_route([], path_id=0, req_id=0, mux="S", swap=[])

    with pytest.raises(ValueError, match="swapping order"):
        ctrl.install_path_on_route(["n1", "n2", "n3", "n4", "n5"], path_id=0, req_id=0, mux="S", swap=[0, 0, 0])

    with pytest.raises(ValueError, match="purif segment r1-r2"):
        ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, req_id=0, mux="S", swap=[1, 0, 1], purif={"r1-r2": 1})

    with pytest.raises(ValueError, match="purif segment n1-n2-n3"):
        ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, req_id=0, mux="S", swap=[1, 0, 1], purif={"n1-n2-n3": 1})

    with pytest.raises(ValueError, match="purif segment n2-n2"):
        ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, req_id=0, mux="S", swap=[1, 0, 1], purif={"n2-n2": 1})

    with pytest.raises(ValueError, match="purif segment n3-n1"):
        ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, req_id=0, mux="S", swap=[1, 0, 1], purif={"n3-n1": 1})


def test_proactive_isolated():
    """Test isolated links mode where swapping is disabled."""
    net, simulator = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, req_id=0, mux="B", swap=[0, 0, 0], m_v=[(1, 1), (1, 1)])
    simulator.run()

    assert f1.e2e_count == 0
    assert f3.e2e_count == 0


def test_proactive_basic():
    """Test basic swapping."""
    net, simulator = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path_on_route(["n1", "n2", "n3"], path_id=0, req_id=0, mux="B", swap=[1, 0, 1], m_v=[(1, 1), (1, 1)])
    simulator.run()

    for app in (f1, f2, f3):
        print((app.own.name, app.e2e_count, app.fidelity / app.e2e_count if app.e2e_count != 0 else None))

    assert f1.e2e_count == f3.e2e_count > 20
    assert f1.fidelity / f1.e2e_count == pytest.approx(f3.fidelity / f3.e2e_count, abs=1e-3)
    assert f1.fidelity / f1.e2e_count >= 0.7
    assert f2.e2e_count == 0


def test_proactive_purif_link1r():
    """Test 1-round purification on each link."""
    net, simulator = build_linear_network(3, qchannel_capacity=2)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path_on_route(
        ["n1", "n2", "n3"], path_id=0, req_id=0, mux="B", swap=[1, 0, 1], purif={"n1-n2": 1, "n2-n3": 1}, m_v=[(2, 2), (2, 2)]
    )
    simulator.run()

    for app in (f1, f2, f3):
        print((app.own.name, app.e2e_count, app.fidelity / app.e2e_count if app.e2e_count != 0 else None))

    assert f1.e2e_count == f3.e2e_count > 10
    assert f1.fidelity / f1.e2e_count == pytest.approx(f3.fidelity / f3.e2e_count, abs=1e-3)
    assert f1.fidelity / f1.e2e_count >= 0.7


def test_proactive_purif_link2r():
    """Test 2-round purification on each link."""
    net, simulator = build_linear_network(3, qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingControllerApp)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path_on_route(
        ["n1", "n2", "n3"], path_id=0, req_id=0, mux="B", swap=[1, 0, 1], purif={"n1-n2": 2, "n2-n3": 2}, m_v=[(4, 4), (4, 4)]
    )
    simulator.run()

    for app in (f1, f2, f3):
        print((app.own.name, app.e2e_count, app.fidelity / app.e2e_count if app.e2e_count != 0 else None))

    assert f1.e2e_count == f3.e2e_count > 10
    assert f1.fidelity / f1.e2e_count == pytest.approx(f3.fidelity / f3.e2e_count, abs=1e-3)
    assert f1.fidelity / f1.e2e_count >= 0.8
