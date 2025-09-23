from mqns.entity.node import Application, QNode
from mqns.entity.operator import OperateRequestEvent, OperateResponseEvent, QuantumOperator
from mqns.models.qubit import H, Qubit
from mqns.simulator import Simulator


def gate_z_and_measure(qubit: Qubit):
    H(qubit=qubit)
    result = qubit.measure()
    return result


def test_operator_sync():
    n1 = QNode("n1")
    o1 = QuantumOperator(name="o1", node=n1, gate=gate_z_and_measure)

    n1.add_operator(o1)

    s = Simulator(0, 10, 1000)
    n1.install(s)

    qubit = Qubit()
    ret = o1.operate(qubit)
    assert ret in [0, 1]

    s.run()


class RecvOperateApp(Application):
    def __init__(self):
        super().__init__()
        self.add_handler(self.OperateResponseEventhandler, OperateResponseEvent)
        self.count = 0

    def OperateResponseEventhandler(self, event: OperateResponseEvent) -> bool | None:
        result = event.result
        assert self.simulator.tc.sec == 0.5
        assert result in [0, 1]
        self.count += 1


def test_operator_async():
    n1 = QNode("n1")
    o1 = QuantumOperator(name="o1", node=n1, gate=gate_z_and_measure, delay=0.5)

    n1.add_operator(o1)
    a1 = RecvOperateApp()
    n1.add_apps(a1)

    s = Simulator(0, 10, 1000)
    n1.install(s)

    qubit = Qubit()
    request = OperateRequestEvent(o1, qubits=[qubit], t=s.time(sec=0), by=n1)
    s.add_event(request)

    s.run()
    assert a1.count == 1
