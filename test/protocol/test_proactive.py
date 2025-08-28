import pytest

from qns.entity import Controller
from qns.network.network import ClassicTopology, QuantumNetwork
from qns.network.proactive import LinkLayer, ProactiveForwarder, ProactiveRoutingController, RoutingPathStatic
from qns.network.proactive.message import MultiplexingVector, validate_path_instructions
from qns.network.topology import LinearTopology
from qns.simulator import Simulator
from qns.utils import log


def build_linear_network(
    n_nodes: int,
    *,
    qchannel_capacity=1,
) -> tuple[QuantumNetwork, Simulator, MultiplexingVector]:
    topo = LinearTopology(
        nodes_number=n_nodes,
        nodes_apps=[LinkLayer(), ProactiveForwarder(ps=0.5)],
        qchannel_args={"length": 100},  # delay is 0.0005 seconds
        cchannel_args={"length": 100},
        memory_args={"decoherence_rate": 1 / 5.0, "capacity": 2 * qchannel_capacity},
    )
    topo.controller = Controller("ctrl", apps=[ProactiveRoutingController()])

    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)
    for qchannel in net.qchannels:
        qchannel.assign_memory_qubits(capacity=qchannel_capacity)
    topo.connect_controller(net.nodes)

    simulator = Simulator(0.0, 60.0)
    log.install(simulator)
    net.install(simulator)

    m_v = [(qchannel_capacity, qchannel_capacity)] * (n_nodes - 1)
    return net, simulator, m_v


def test_path_validation():
    """Test path validation logic."""

    route3 = ["n1", "n2", "n3"]
    swap3 = [1, 0, 1]
    mv3 = [(1, 1)] * 2

    with pytest.raises(ValueError, match="swapping order"):
        validate_path_instructions({"req_id": 0, "route": [], "swap": [], "purif": {}})

    with pytest.raises(ValueError, match="swapping order"):
        validate_path_instructions({"req_id": 0, "route": ["n1", "n2", "n3", "n4", "n5"], "swap": [0, 0, 0], "purif": {}})

    with pytest.raises(ValueError, match="multiplexing vector"):
        validate_path_instructions({"req_id": 0, "route": route3, "swap": swap3, "m_v": [(1, 1)] * 3, "purif": {}})

    with pytest.raises(ValueError, match="purif segment r1-r2"):
        validate_path_instructions({"req_id": 0, "route": route3, "swap": swap3, "m_v": mv3, "purif": {"r1-r2": 1}})

    with pytest.raises(ValueError, match="purif segment n1-n2-n3"):
        validate_path_instructions({"req_id": 0, "route": route3, "swap": swap3, "m_v": mv3, "purif": {"n1-n2-n3": 1}})

    with pytest.raises(ValueError, match="purif segment n2-n2"):
        validate_path_instructions({"req_id": 0, "route": route3, "swap": swap3, "m_v": mv3, "purif": {"n2-n2": 1}})

    with pytest.raises(ValueError, match="purif segment n3-n1"):
        validate_path_instructions({"req_id": 0, "route": route3, "swap": swap3, "m_v": mv3, "purif": {"n3-n1": 1}})


def test_proactive_isolated():
    """Test isolated links mode where swapping is disabled."""
    net, simulator, m_v = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3"], swap=[0, 0, 0], m_v=m_v))
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
    net, simulator, m_v = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], m_v=m_v))
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
    net, simulator, m_v = build_linear_network(4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3", "n4"], swap=[1, 0, 0, 1], m_v=m_v))
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
    net, simulator, m_v = build_linear_network(3, qchannel_capacity=2)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], m_v=m_v, purif={"n1-n2": 1, "n2-n3": 1}))
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
    net, simulator, m_v = build_linear_network(3, qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], m_v=m_v, purif={"n1-n2": 2, "n2-n3": 2}))
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
    net, simulator, m_v = build_linear_network(3, qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], m_v=m_v, purif={"n1-n3": 2}))
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
