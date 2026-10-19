import itertools
from typing import cast

import numpy as np
import pytest

from mqns.entity.node import Application, Controller, Node, NodePair, QNode, split_node_pair
from mqns.models.core import TrafficMatrixInput
from mqns.network.network import (
    MatrixTrafficGenerator,
    QuantumNetwork,
    RequestActiveEvent,
    RequestInactiveEvent,
    RequestState,
    TimingModeSync,
    TimingPhase,
    TrafficMatrixMapping,
    sync_phase_handler,
)
from mqns.simulator import Simulator, Time, event_handler, func_to_event


class SyncCheckApp(Application[Node]):
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    @sync_phase_handler(TimingPhase.EXTERNAL, True)
    def sync_external_enter(self) -> None:
        assert self.simulator.tc.slot % 5000 == 0
        self.records.append("E+")

    @sync_phase_handler(TimingPhase.EXTERNAL, False)
    def sync_external_exit(self) -> None:
        assert self.simulator.tc.slot % 5000 == 4000
        self.records.append("E-")

    @sync_phase_handler(TimingPhase.INTERNAL, True)
    def sync_internal_enter(self) -> None:
        assert self.simulator.tc.slot % 5000 == 4000
        self.records.append("I+")

    @sync_phase_handler(TimingPhase.INTERNAL, False)
    def sync_internal_exit(self) -> None:
        assert self.simulator.tc.slot % 5000 == 0
        self.records.append("I-")


def test_timing_mode_sync():
    timing = TimingModeSync(t_ext=4, t_int=1)
    net = QuantumNetwork(timing=timing)
    net.add_node(QNode("n1", apps=[SyncCheckApp()]))
    net.add_node(QNode("n2", apps=[SyncCheckApp()]))
    net.set_controller(Controller("ctrl", apps=[SyncCheckApp()]))

    s = Simulator(0.0, 29.9, accuracy=1000, install_to=(net,))
    assert set(node.name for node in net.all_nodes) == {"ctrl", "n1", "n2"}
    assert timing.t_ext.slot == 4000
    assert timing.t_rtg.slot == 0
    assert timing.t_int.slot == 1000

    s.run()

    for node in net.all_nodes:
        app = node.get_app(SyncCheckApp)
        assert app.records == ["E+", "E-", "I+", "I-"] * 5 + ["E+", "E-", "I+"]


@pytest.mark.parametrize(
    ("tm", "src_dst"),
    [
        pytest.param([[0, 1, 0], [0, 0, 0], [0, 0, 0]], "A-B", id="list[list]"),
        pytest.param(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), "B-C", id="ndarray"),
        pytest.param({"C-A": 1}, "C-A", id="dict"),
    ],
)
def test_mtg_eager(tm: TrafficMatrixInput | TrafficMatrixMapping, src_dst: NodePair):
    """
    Test MatrixTrafficGenerator with sched=eager.
    Also validate each traffic matrix input format; the inputs are one-hot.
    """
    net = QuantumNetwork()
    net.add_node(QNode("A"))
    net.add_node(QNode("B"))
    net.add_node(QNode("C"))

    mtg = MatrixTrafficGenerator(net, tm, rate=10, sched="eager")
    Simulator(100, 200, install_to=(net, mtg))

    # expect 1000 requests in 100 seconds for 10Hz
    assert 800 <= len(net.requests) <= 1200

    src_dst = split_node_pair(src_dst)
    for req in net.requests:
        assert req.state is RequestState.NEW
        assert (req.src, req.dst) == src_dst

    for req0, req1 in itertools.pairwise(net.requests):
        assert req0.active_since <= req1.active_until


class RequestCheckApp(Application[Controller]):
    net: QuantumNetwork

    def __init__(self):
        super().__init__()
        self.enters: list[int] = []
        """Length of ``net.requests`` upon each ``RequestActiveEvent``."""
        self.exits = 0
        """Count of ``RequestInactiveEvent``."""

    @event_handler
    def handle_request_active(self, event: RequestActiveEvent) -> None:
        req = event.req
        assert (req.src, req.dst) == ("A", "C")
        assert (cast(Time, req.active_until) - cast(Time, req.active_since)).sec == pytest.approx(0.1, abs=1e-6)
        assert req.epr_count == 2
        assert req.state is RequestState.ACTIVE
        self.enters.append(len(self.net.requests))

    @event_handler
    def handle_request_inactive(self, event: RequestInactiveEvent) -> None:
        req = event.req
        assert (req.src, req.dst) == ("A", "C")
        assert req.state is RequestState.EXPIRED
        self.exits += 1


def test_mtg_lazy():
    """
    Test MatrixTrafficGenerator with sched=lazy.
    Also validate request attributes.
    """
    net = QuantumNetwork()
    net.add_node(QNode("A"))
    net.add_node(QNode("B"))
    net.add_node(QNode("C"))
    net.set_controller(Controller("ctrl", apps=[app := RequestCheckApp()]))
    app.net = net

    mtg = MatrixTrafficGenerator(
        net,
        {"A-C": 1},
        rate=10,
        sched="lazy",
        duration=0.1,
        epr_count=2,
    )
    s = Simulator(100, np.inf, need_synchronized=False, install_to=(net, mtg))
    s.sched(func_to_event(s.time(sec=200), s.stop))
    s.run()

    # expect 1000 requests in 100 seconds for 10Hz
    assert 800 <= len(app.enters) <= 1200

    # no more than 100 extra requests upon entering each request
    for i, n_requests in enumerate(app.enters):
        assert n_requests <= i + 100
