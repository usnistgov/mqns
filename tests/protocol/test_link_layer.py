from typing import override

import pytest

from mqns.entity.memory import QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import LinkArchAlways, LinkArchDimBk
from mqns.models.epr import (
    Entanglement,
    MixedStateEntanglement,
    WernerStateEntanglement,
)
from mqns.network.network import QuantumNetwork, TimingModeSync
from mqns.network.protocol.event import (
    ManageActiveChannels,
    QubitDecoheredEvent,
    QubitEntangledEvent,
    QubitReleasedEvent,
)
from mqns.network.protocol.link_layer import LinkLayer, LinkLayerCounters
from mqns.network.topology import ClassicTopology, CustomTopology, LinearTopology
from mqns.simulator import Simulator
from mqns.utils import log


class NetworkLayer(Application[QNode]):
    def __init__(self):
        super().__init__()
        self.release_after: float | None = None
        """If set, ``QubitReleasedEvent`` would be emitted after specified duration for the next entanglement."""
        self.entangle: list[tuple[float, float]] = []
        """Entanglement events, each entry contains entanglement time and EPR creation time."""
        self.decohere: list[float] = []
        """Decoherence events, each entry is event time."""

        self.add_handler(self.handle_entangle, QubitEntangledEvent)
        self.add_handler(self.handle_decohere, QubitDecoheredEvent)

    @override
    def install(self, node):
        self._application_install(node, QNode)
        self.memory = self.node.memory
        self.epr_type = self.node.network.epr_type

    def handle_entangle(self, event: QubitEntangledEvent):
        qubit, epr = self.memory.read(event.qubit.addr, has=self.epr_type)
        assert qubit is event.qubit
        t_create = epr.decohere_time - self.memory.decoherence_delay
        self.entangle.append((event.t.sec, t_create.sec))

        if not isinstance(self.release_after, float):
            return
        self.memory.read(event.qubit.addr, remove=True)
        event.qubit.state = QubitState.RELEASE
        self.simulator.add_event(QubitReleasedEvent(self.node, event.qubit, t=event.t + self.release_after, by=self))
        self.release_after = None

    def handle_decohere(self, event: QubitDecoheredEvent):
        self.decohere.append(event.t.sec)


def manage_active_channel(simulator: Simulator, t: float, src: NetworkLayer, dst: NetworkLayer, *, stop=False):
    simulator.add_event(
        ManageActiveChannels(
            src.node,
            dst.node,
            src.node.get_qchannel(dst.node),
            start=not stop,
            t=simulator.time(sec=t),
            by=src,
        )
    )


@pytest.mark.parametrize(
    "epr_type",
    [
        WernerStateEntanglement,
        MixedStateEntanglement,
    ],
)
def test_basic(epr_type: type[Entanglement]):
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"delay": 0.1, "link_arch": LinkArchAlways(LinkArchDimBk())},
        cchannel_args={"delay": 0.1},
        memory_args={"t_cohere": 4.1},
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow, epr_type=epr_type)
    net.build_route()
    net.get_qchannel("n1", "n2").assign_memory_qubits(capacity=1)

    s = Simulator(0.0, 20.0, install_to=(log, net))

    ll1 = net.get_node("n1").get_app(LinkLayer)
    ll2 = net.get_node("n2").get_app(LinkLayer)
    nl1 = net.get_node("n1").get_app(NetworkLayer)
    nl2 = net.get_node("n2").get_app(NetworkLayer)
    manage_active_channel(s, 0.5, nl1, nl2)
    manage_active_channel(s, 8.4, nl1, nl2, stop=True)
    nl1.release_after = 2.9
    nl2.release_after = 3.2

    s.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        print(ll.node.name, ll.cnt, nl.entangle, nl.decohere)
        assert len(nl.entangle) == 3
        assert len(nl.decohere) == 2
        # t=0.5, n1 installs path
        # t=0.5, n1 sends RESERVE_QUBIT
        # t=0.6, n2 receives RESERVE_QUBIT and sends RESERVE_QUBIT_OK
        # t=0.7, n1 receives RESERVE_QUBIT_OK
        # t=0.9, entanglement established
        # t=0.7 is assumed time of entanglement creation
        assert nl.entangle[0] == pytest.approx((0.9, 0.7), abs=1e-3)
        # t=3.8, n1 releases qubit and sends RESERVE_QUBIT
        # t=3.9, n2 receives RESERVE_QUBIT but has no qubit available
        # t=4.1, n2 releases qubit and sends RESERVE_QUBIT_OK
        # t=4.2, n1 receives RESERVE_QUBIT_OK
        # t=4.4, entanglement established
        # t=4.2 is assumed time of entanglement creation
        assert nl.entangle[1] == pytest.approx((4.4, 4.2), abs=1e-3)
        # t=8.3, qubits decohered 4.1 seconds since entanglement creation
        assert nl.decohere[0] == pytest.approx(8.3, abs=1e-3)
        # t=8.3, n1 sends RESERVE_QUBIT
        # t=8.4, n1 uninstalls path, but this does not affect ongoing reservation
        # t=8.4, n2 receives RESERVE_QUBIT and sends RESERVE_QUBIT_OK
        # t=8.5, n1 receives RESERVE_QUBIT_OK
        # t=8.7, entanglement established
        # t=8.5 is assumed time of entanglement creation
        assert nl.entangle[2] == pytest.approx((8.7, 8.5), abs=1e-3)
        # t=12.6, qubits decohered 4.1 seconds since entanglement creation
        assert nl.decohere[1] == pytest.approx(12.6, abs=1e-3)
        # no more entanglements because the path has been uninstalled

    ll_cnt_agg = LinkLayerCounters.aggregate(net.nodes)
    print("ll_cnt_agg", ll_cnt_agg)
    assert ll_cnt_agg.n_etg == 3
    assert ll_cnt_agg.n_attempts == 3
    # n_decoh is only incremented on the primary node.
    # Although there are two decoherence events in the simulation, l1.n_decoh is incremented only at t=8.3.
    # For the t=12.6 event, the path is uninstalled, so that l1 cannot recognize itself as primary.
    assert ll_cnt_agg.n_decoh == 1
    assert ll_cnt_agg.decoh_ratio == pytest.approx(1 / 3, abs=1e-6)


def test_skip_ahead():
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"length": 100},
        cchannel_args={"length": 100},
        memory_args={"t_cohere": 1.0},
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow)
    net.build_route()
    net.get_qchannel("n1", "n2").assign_memory_qubits(capacity=1)

    simulator = Simulator(0.0, 10.0, install_to=(log, net))

    ll1 = net.get_node("n1").get_app(LinkLayer)
    ll2 = net.get_node("n2").get_app(LinkLayer)
    nl1 = net.get_node("n1").get_app(NetworkLayer)
    nl2 = net.get_node("n2").get_app(NetworkLayer)
    manage_active_channel(simulator, 0.5, nl1, nl2)

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        print(ll.node.name, ll.cnt, nl.entangle, nl.decohere)

    assert len(nl1.entangle) == len(nl2.entangle) > 0
    for t1, t2 in zip(nl1.entangle, nl2.entangle):
        assert t1 == pytest.approx(t2, abs=1e-3)

    ll_cnt_agg = LinkLayerCounters.aggregate(net.nodes)
    print("ll_cnt_agg", ll_cnt_agg)
    assert 0 <= ll_cnt_agg.decoh_ratio <= 1


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
        memory_args={"t_cohere": 10.0},
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow, timing=TimingModeSync(t_ext=0.6, t_int=0.4))
    net.build_route()

    simulator = Simulator(0.0, 10.0, install_to=(log, net))

    nl0 = net.get_node("n0").get_app(NetworkLayer)
    nl1 = net.get_node("n1").get_app(NetworkLayer)
    nl2 = net.get_node("n2").get_app(NetworkLayer)
    nl3 = net.get_node("n3").get_app(NetworkLayer)
    manage_active_channel(simulator, 0.1, nl0, nl1)
    manage_active_channel(simulator, 0.1, nl2, nl3)  # n=1, start entanglements
    manage_active_channel(simulator, 1.1, nl2, nl3)  # n=2, no change
    manage_active_channel(simulator, 4.1, nl2, nl3, stop=True)  # n=1, no change
    manage_active_channel(simulator, 5.1, nl2, nl3, stop=True)  # n=0, stop entanglements

    simulator.run()

    for nl in nl0, nl1:
        print(nl.node.name, nl.entangle, nl.decohere)
        # τ=0.2 for the channel between n0 and n1.
        # Entanglement (including reservation) requires 4τ i.e. 0.8 seconds but the EXTERNAL phase
        # has only 0.6 seconds, so that no entanglement could complete on this channel.
        assert len(nl.entangle) == 0
        assert len(nl.decohere) == 0

    for nl in nl2, nl3:
        print(nl.node.name, nl.entangle, nl.decohere)
        # τ=0.1 for the channel between n2 and n3.
        # Entanglement (including reservation) requires 4τ i.e. 0.4 seconds.
        # No entanglement occurs in the first EXTERNAL phase window, because reservations are only initiated
        # at the start of each EXTERNAL phase window, not when ManageActiveChannels arrives.
        # The uninstall_path stop event takes effect for the EXTERNAL phase window starting at t=6.0.
        assert [t_notify for t_notify, _ in nl.entangle] == pytest.approx([1.4, 2.4, 3.4, 4.4, 5.4], abs=1e-3)
        # All qubits are cleared at the start of each EXTERNAL phase, before memory decoherence occurs.
        # Decoherence events are not emitted for cleared qubits.
        assert len(nl.decohere) == 0
