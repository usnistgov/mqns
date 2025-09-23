import pytest

from mqns.entity.memory import QubitState
from mqns.entity.node import Application, Node, QNode
from mqns.entity.qchannel import LinkArchAlways, LinkArchDimBk
from mqns.models.epr import BaseEntanglement
from mqns.network.network import QuantumNetwork, TimingModeSync
from mqns.network.protocol.event import (
    ManageActiveChannels,
    QubitDecoheredEvent,
    QubitEntangledEvent,
    QubitReleasedEvent,
)
from mqns.network.protocol.link_layer import LinkLayer
from mqns.network.topology import ClassicTopology, CustomTopology, LinearTopology
from mqns.simulator import Simulator
from mqns.utils import log


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
        qubit, epr = self.memory.read(event.qubit.addr, must=True, destructive=False)
        assert qubit == event.qubit
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


def manage_active_channel(simulator: Simulator, t: float, src: NetworkLayer, dst: NetworkLayer, *, stop=False):
    simulator.add_event(
        ManageActiveChannels(
            src.own,
            dst.own,
            src.own.get_qchannel(dst.own),
            start=not stop,
            t=simulator.time(sec=t),
            by=src,
        )
    )


def test_basic():
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"delay": 0.1, "link_arch": LinkArchAlways(LinkArchDimBk())},
        cchannel_args={"delay": 0.1},
        memory_args={"decoherence_rate": 1 / 4.1},
    )
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)
    net.build_route()
    net.get_qchannel("n1", "n2").assign_memory_qubits(capacity=1)

    simulator = Simulator(0.0, 20.0)
    log.install(simulator)
    net.install(simulator)

    a1 = net.get_node("n1").get_app(NetworkLayer)
    a2 = net.get_node("n2").get_app(NetworkLayer)
    manage_active_channel(simulator, 0.5, a1, a2)
    manage_active_channel(simulator, 8.4, a1, a2, stop=True)
    a1.release_after = 2.9
    a2.release_after = 3.2

    simulator.run()

    for app in (a1, a2):
        print(app.own.name, app.entangle, app.decohere)
        assert len(app.entangle) == 3
        assert len(app.decohere) == 2
        # t=0.5, n1 installs path
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
        # t=8.4, n1 uninstalls path, but this does not affect ongoing reservation
        # t=8.4, n2 receives RESERVE_QUBIT and sends RESERVE_QUBIT_OK
        # t=8.5, n1 receives RESERVE_QUBIT_OK
        # t=8.7, entanglement established
        # t=8.5 is assumed time of entanglement creation
        assert app.entangle[2] == pytest.approx((8.7, 8.5), abs=1e-3)
        # t=12.6, qubits decohered 4.1 seconds since entanglement creation
        assert app.decohere[1] == pytest.approx(12.6, abs=1e-3)
        # no more entanglements because the path has been uninstalled


def test_skip_ahead():
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"length": 100},
        cchannel_args={"length": 100},
        memory_args={"decoherence_rate": 1 / 1.0},
    )
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)
    net.build_route()
    net.get_qchannel("n1", "n2").assign_memory_qubits(capacity=1)

    simulator = Simulator(0.0, 10.0)
    log.install(simulator)
    net.install(simulator)

    a1 = net.get_node("n1").get_app(NetworkLayer)
    a2 = net.get_node("n2").get_app(NetworkLayer)
    manage_active_channel(simulator, 0.5, a1, a2)

    simulator.run()

    for app in (a1, a2):
        print(app.own.name, app.entangle, app.decohere)

    assert len(a1.entangle) == len(a2.entangle) > 0
    for t1, t2 in zip(a1.entangle, a2.entangle):
        assert t1 == pytest.approx(t2, abs=1e-3)


def test_timing_mode_sync():
    topo = CustomTopology(
        {
            "qnodes": [
                {"name": "n0"},
                {"name": "n1"},
                {"name": "n2"},
                {"name": "n3"},
            ],
            "qchannels": [
                {"node1": "n0", "node2": "n1", "parameters": {"delay": 0.2, "link_arch": LinkArchAlways(LinkArchDimBk())}},
                {"node1": "n2", "node2": "n3", "parameters": {"delay": 0.1, "link_arch": LinkArchAlways(LinkArchDimBk())}},
            ],
        },
        nodes_apps=[NetworkLayer(), LinkLayer()],
        memory_args={"decoherence_rate": 1 / 10.0},
    )
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow, timing=TimingModeSync(t_ext=0.6, t_int=0.4))
    net.build_route()

    simulator = Simulator(0.0, 10.0)
    log.install(simulator)
    net.install(simulator)

    a0 = net.get_node("n0").get_app(NetworkLayer)
    a1 = net.get_node("n1").get_app(NetworkLayer)
    a2 = net.get_node("n2").get_app(NetworkLayer)
    a3 = net.get_node("n3").get_app(NetworkLayer)
    manage_active_channel(simulator, 0.1, a0, a1)
    manage_active_channel(simulator, 0.1, a2, a3)  # n=1, start entanglements
    manage_active_channel(simulator, 1.1, a2, a3)  # n=2, no change
    manage_active_channel(simulator, 4.1, a2, a3, stop=True)  # n=1, no change
    manage_active_channel(simulator, 5.1, a2, a3, stop=True)  # n=0, stop entanglements

    simulator.run()

    for app in (a0, a1):
        print(app.own.name, app.entangle, app.decohere)
        # τ=0.2 for the channel between n0 and n1.
        # Entanglement (including reservation) requires 4τ i.e. 0.8 seconds but the EXTERNAL phase
        # has only 0.6 seconds, so that no entanglement could complete on this channel.
        assert len(app.entangle) == 0
        assert len(app.decohere) == 0

    for app in (a2, a3):
        print(app.own.name, app.entangle, app.decohere)
        # τ=0.1 for the channel between n2 and n3.
        # Entanglement (including reservation) requires 4τ i.e. 0.4 seconds.
        # No entanglement occurs in the first EXTERNAL phase window, because reservations are only initiated
        # at the start of each EXTERNAL phase window, not when ManageActiveChannels arrives.
        # The uninstall_path stop event takes effect for the EXTERNAL phase window starting at t=6.0.
        assert [t_notify for t_notify, _ in app.entangle] == pytest.approx([1.4, 2.4, 3.4, 4.4, 5.4], abs=1e-3)
        # All qubits are cleared at the start of each EXTERNAL phase, before memory decoherence occurs.
        # Decoherence events are not emitted for cleared qubits.
        assert len(app.decohere) == 0
