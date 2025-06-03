
from qns.entity.node import Application, QNode
from qns.entity.qchannel import QuantumChannel, QubitLossChannel, RecvQubitPacket
from qns.models.qubit import Qubit
from qns.simulator import Event, Simulator, func_to_event


class SendApp(Application):
    def __init__(self, dest: QNode, qchannel: QuantumChannel, send_interval: float = 1):
        super().__init__()
        self.dest = dest
        self.qchannel = qchannel
        self.send_interval = send_interval
        self.count = 0

    def install(self, node, simulator: Simulator):
        super().install(node=node, simulator=simulator)
        t = self.simulator.ts
        event = func_to_event(t, self.send, by=self)
        self.simulator.add_event(event)

    def send(self):
        self.qchannel.send(qubit=Qubit(), next_hop=self.dest)
        self.count += 1

        t = self.simulator.current_time + self.send_interval
        event = func_to_event(t, self.send, by=self)
        self.simulator.add_event(event)


class RecvApp(Application):
    def __init__(self):
        super().__init__()
        self.add_handler(self.RecvQubitHandler, [RecvQubitPacket])
        self.count = 0

    def RecvQubitHandler(self, node, event: Event) -> bool|None:
        self.count += 1


def setup_and_run(l1: QuantumChannel) -> tuple[SendApp, RecvApp]:
    n1 = QNode(name="n_1")
    n2 = QNode(name="n_2")
    n1.add_qchannel(l1)
    n2.add_qchannel(l1)

    a1 = SendApp(dest=n2, qchannel=l1, send_interval=0.010)
    n1.add_apps(a1)
    a2 = RecvApp()
    n2.add_apps(a2)

    s = Simulator(1.000, 4.999, 1000)
    n1.install(s)
    n2.install(s)
    s.run()

    return (a1, a2)

def test_qchannel_perfect():
    l1 = QuantumChannel()
    a1, a2 = setup_and_run(l1)
    assert a1.count == 400
    assert a2.count == 400

def test_qchannel_delay():
    l1 = QuantumChannel(delay=0.100)
    a1, a2 = setup_and_run(l1)
    assert a1.count == 400
    assert a2.count == 390

def test_qchannel_drop():
    l1 = QuantumChannel(drop_rate=0.1)
    a1, a2 = setup_and_run(l1)
    assert a1.count == 400
    assert 320 < a2.count < 400

def test_qchannel_bandwidth():
    l1 = QuantumChannel(bandwidth=10, max_buffer_size=5)
    a1, a2 = setup_and_run(l1)
    assert a1.count == 400
    assert a2.count == 40

def test_qubit_loss_channel():
    l1 = QubitLossChannel(name="loss_channel_1", p_init=0.1, attenuation_rate=0.02, length=100)
    a1, a2 = setup_and_run(l1)
    print(l1.drop_rate, a1.count, a2.count)
