import pytest

from qns.entity.memory import QubitState
from qns.entity.node import Application, Node, QNode
from qns.entity.qchannel import LinkArchDimBk
from qns.models.epr import BaseEntanglement
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
from qns.utils import log

try:
    from typing import override
except ImportError:
    from typing_extensions import override


class NetworkLayer(Application):
    def __init__(self):
        super().__init__()
        self.release_after: float | None = None
        self.entangle: list[tuple[float, float]] = []
        self.decohere: list[float] = []

        self.add_handler(self.handle_entangle, QubitEntangledEvent)
        self.add_handler(self.handle_decohere, QubitDecoheredEvent)

    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        self.own = self.get_node(node_type=QNode)
        self.memory = self.own.get_memory()

    def handle_entangle(self, event: QubitEntangledEvent):
        _, epr = self.memory.read(event.qubit.addr, must=True, destructive=False)
        assert isinstance(epr, BaseEntanglement)
        assert epr.creation_time is not None
        self.entangle.append((event.t.sec, epr.creation_time.sec))

        if not isinstance(self.release_after, float):
            return
        self.memory.read(event.qubit.addr)
        event.qubit.state = QubitState.RELEASE
        self.simulator.add_event(QubitReleasedEvent(self.own, event.qubit, t=event.t + self.release_after, by=self))
        self.release_after = None

    def handle_decohere(self, event: QubitDecoheredEvent):
        self.decohere.append(event.t.sec)


class LinkArchDimBkAlways(LinkArchDimBk):
    def __init__(self, name="DIM-BK-always"):
        super().__init__(name)

    @override
    def success_prob(self, **_) -> float:
        return 1.0


def test_link_layer_basic():
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"delay": 0.1, "link_arch": LinkArchDimBkAlways()},
        cchannel_args={"delay": 0.1},
        memory_args={"decoherence_rate": 1 / 4.1},
    )
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)
    net.build_route()
    n1 = net.get_node("n1")
    n2 = net.get_node("n2")
    net.get_qchannel("l0,1").assign_memory_qubits(capacity=1)

    simulator = Simulator(0.0, 10.0)
    log.install(simulator)
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
        # t=0.6, n2 receives RESERVE_QUBIT and sends RESERVE_QUBIT_OK
        # t=0.7, n1 receives RESERVE_QUBIT_OK
        # t=0.9, entanglement established
        # t=0.7 is assumed time of entanglement creation
        assert app.entangle[0] == pytest.approx((0.9, 0.7), abs=1e-3)
        # t=3.8, n1 releases qubit and sends RESERVE_QUBIT
        # t=3.9, n2 receives RESERVE_QUBIT but has no qubit available
        # t=4.1, n2 releases qubit and sends RESERVE_QUBIT_OK
        # t=4.2, n1 receives RESERVE_QUBIT_OK
        # t=4.4, entanglement established
        # t=4.2 is assumed time of entanglement creation
        assert app.entangle[1] == pytest.approx((4.4, 4.2), abs=1e-3)
        # t=8.3, qubits decohered 4.1 seconds since entanglement creation
        assert app.decohere[0] == pytest.approx(8.3, abs=1e-3)
        # t=8.3, n1 sends RESERVE_QUBIT
        # t=8.4, n2 receives RESERVE_QUBIT and sends RESERVE_QUBIT_OK
        # t=8.5, n1 receives RESERVE_QUBIT_OK
        # t=8.7, entanglement established
        # t=8.5 is assumed time of entanglement creation
        assert app.entangle[2] == pytest.approx((8.7, 8.5), abs=1e-3)


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
