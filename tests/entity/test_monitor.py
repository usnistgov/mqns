import math
from typing import override

from mqns.entity.monitor import Monitor
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import QuantumChannel, RecvQubitPacket
from mqns.models.qubit import Qubit
from mqns.simulator import Simulator, func_to_event


class SendApp(Application[QNode]):
    def __init__(self, dest: QNode, qchannel: QuantumChannel, send_rate=1):
        super().__init__()
        self.dest = dest
        self.qchannel = qchannel
        self.send_rate = send_rate
        self.count = 0

    @override
    def install(self, node):
        self._application_install(node, QNode)
        self.simulator.add_event(func_to_event(self.simulator.ts, self.send, by=self))

    def send(self):
        qubit = Qubit()
        self.qchannel.send(qubit, next_hop=self.dest)
        self.count += 1
        self.simulator.add_event(func_to_event(self.simulator.tc + 1 / self.send_rate, self.send, by=self))


class RecvApp(Application[QNode]):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.add_handler(self.RecvQubitHandler, RecvQubitPacket)

    def RecvQubitHandler(self, _: RecvQubitPacket) -> bool | None:
        self.count += 1


def build_network(simulator: Simulator) -> tuple[SendApp, RecvApp]:
    n1 = QNode(name="n_1")
    n2 = QNode(name="n_2")

    l1 = QuantumChannel(name="l1", bandwidth=5, delay=0.5, drop_rate=0.2, max_buffer_size=5)
    n1.add_qchannel(l1)
    n2.add_qchannel(l1)

    sp = SendApp(dest=n2, qchannel=l1, send_rate=3)
    n1.add_apps(sp)
    rp = RecvApp()
    n2.add_apps(rp)

    n1.install(simulator)
    n2.install(simulator)

    return sp, rp


def make_monitor(
    simulator: Simulator, sp: SendApp, rp: RecvApp, *, enable_attributions=True, enable_timed=True, enable_event=True
) -> Monitor:
    m = Monitor("m")

    if enable_attributions:
        m.add_attribution("send_count", lambda s, n, e: sp.count)
        m.add_attribution("recv_count", lambda s, n, e: rp.count)
        m.add_attribution("event_name", lambda s, n, e: e.__class__.__name__ if e.name is None else e.name)

    if enable_timed:
        m.at_start()
        m.at_finish()
        m.at_period(period_time=1)

    if enable_event:
        m.at_event(RecvQubitPacket)

    m.install(simulator)
    return m


def test_monitor_empty():
    simulator = Simulator(0, 10, accuracy=1000)
    sp, rp = build_network(simulator)
    m = make_monitor(simulator, sp, rp, enable_timed=False, enable_event=False)

    simulator.run()

    data = m.get_data()
    print(data)

    assert len(data) == 0


def test_monitor_time_only():
    simulator = Simulator(0, 10, accuracy=1000)
    sp, rp = build_network(simulator)
    m = make_monitor(simulator, sp, rp, enable_attributions=False)

    simulator.run()

    data = m.get_data()
    print(data)

    assert set(data.columns) == set(["time"])
    assert data.at[0, "time"] == 0
    assert data.at[data.shape[0] - 1, "time"] == 10


def test_monitor_full_finite():
    simulator = Simulator(0, 10, accuracy=1000)
    sp, rp = build_network(simulator)
    m = make_monitor(simulator, sp, rp)

    simulator.run()

    data = m.get_data()
    print(data)

    assert set(data.columns) == set(["time", "send_count", "recv_count", "event_name"])

    count_by_event_name = data.value_counts(subset="event_name")
    assert count_by_event_name.shape[0] == 4
    assert count_by_event_name["RecvQubitPacket"] > 0
    assert count_by_event_name["start watch event"] == 1
    assert count_by_event_name["finish watch event"] == 1
    assert count_by_event_name["period watch event(1)"] == 11


def test_monitor_full_continuous():
    simulator = Simulator(0, math.inf, accuracy=1000)
    sp, rp = build_network(simulator)
    m = make_monitor(simulator, sp, rp)

    simulator.add_event(func_to_event(simulator.time(sec=9.5), simulator.stop))
    simulator.run()

    data = m.get_data()
    print(data)

    assert set(data.columns) == set(["time", "send_count", "recv_count", "event_name"])

    count_by_event_name = data.value_counts(subset="event_name")
    assert count_by_event_name.shape[0] == 3
    assert count_by_event_name["RecvQubitPacket"] > 0
    assert count_by_event_name["start watch event"] == 1
    assert count_by_event_name["period watch event(1)"] == 10
