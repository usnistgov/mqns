import pytest

from qns.entity import Application, Node, QNode
from qns.network.network import ClassicTopology, QuantumNetwork
from qns.network.protocol.event import (
    ManageActiveChannels,
    QubitDecoheredEvent,
    QubitEntangledEvent,
    QubitReleasedEvent,
    TypeEnum,
)
from qns.network.protocol.link_layer import LinkLayer
from qns.network.topology import LinearTopology
from qns.simulator import Simulator


class NetworkLayer(Application):
    def __init__(self):
        super().__init__()
        self.release_after: float | None = None
        self.entangle: list[float] = []
        self.decohere: list[float] = []

        self.add_handler(self.handle_entangle, QubitEntangledEvent)
        self.add_handler(self.handle_decohere, QubitDecoheredEvent)

    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        self.own = self.get_node(node_type=QNode)

    def handle_entangle(self, event: QubitEntangledEvent):
        self.entangle.append(event.t.sec)
        if not isinstance(self.release_after, float):
            return
        self.own.get_memory().read(address=event.qubit.addr)
        event.qubit.fsm.to_release()
        self.simulator.add_event(QubitReleasedEvent(self.own, event.qubit, t=event.t + self.release_after, by=self))
        self.release_after = None

    def handle_decohere(self, event: QubitDecoheredEvent):
        self.decohere.append(event.t.sec)


def test_link_layer_basic():
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"delay": 0.1},
        cchannel_args={"delay": 0.2},
        memory_args={"decoherence_rate": 1 / 4.1},
    )
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)
    net.build_route()
    n1 = net.get_node("n1")
    n2 = net.get_node("n2")
    net.get_qchannel("l0,1").assign_memory_qubits(capacity=1)

    simulator = Simulator(0.0, 10.0)
    net.install(simulator)

    a1 = n1.get_app(NetworkLayer)
    a2 = n2.get_app(NetworkLayer)
    simulator.add_event(
        ManageActiveChannels(
            n1,
            n2,
            TypeEnum.ADD,
            t=simulator.time(sec=0.5),
            by=a1,
        )
    )
    a1.release_after = 2.9
    a2.release_after = 3.2

    simulator.run()

    for app in (a1, a2):
        print((app.get_node().name, app.entangle, app.decohere))
        assert len(app.entangle) == 3
        assert len(app.decohere) == 1
        # t=0.5, n1 sends RESERVE_QUBIT
        # t=0.7, n2 receives RESERVE_QUBIT and sends RESERVE_QUBIT_OK
        # t=0.9, n1 receives RESERVE_QUBIT_OK and sends qubit
        # t=1.0, n2 receives qubit, entanglement established
        assert app.entangle[0] == pytest.approx(1.0, abs=1e-3)
        # t=3.9, n1 releases qubit and sends RESERVE_QUBIT
        # t=4.0, n2 receives RESERVE_QUBIT but has no qubit available
        # t=4.2, n2 releases qubit and sends RESERVE_QUBIT_OK
        # t=4.4, n1 receives RESERVE_QUBIT_OK and sends qubit
        # t=4.5, n2 receives qubit, entanglement established
        # t=4.0 is assumed time of entanglement creation, 4x qchannel.delay prior to sending qubit
        assert app.entangle[1] == pytest.approx(4.5, abs=1e-3)
        # t=8.1, qubits decohered 4.1 seconds since entanglement creation
        assert app.decohere[0] == pytest.approx(8.1, abs=1e-3)
        # t=8.6, entanglement established after 0.5 seconds
        assert app.entangle[2] == pytest.approx(8.6, abs=1e-3)


def test_link_layer_skip_ahead():
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"length": 100},
        cchannel_args={"length": 100},
        memory_args={"decoherence_rate": 1 / 1.0},
    )
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)
    net.build_route()
    n1 = net.get_node("n1")
    n2 = net.get_node("n2")
    qc = net.get_qchannel("l0,1")
    qc.assign_memory_qubits(capacity=1)

    simulator = Simulator(0.0, 10.0)
    net.install(simulator)

    a1 = n1.get_app(NetworkLayer)
    a2 = n2.get_app(NetworkLayer)
    simulator.add_event(
        ManageActiveChannels(
            n1,
            n2,
            TypeEnum.ADD,
            t=simulator.time(sec=0.5),
            by=a1,
        )
    )

    simulator.run()

    for app in (a1, a2):
        print((app.get_node().name, app.entangle, app.decohere))

    assert len(a1.entangle) == len(a2.entangle) > 0
    for t1, t2 in zip(a1.entangle, a2.entangle):
        assert t1 == pytest.approx(t2, abs=1e-3)
