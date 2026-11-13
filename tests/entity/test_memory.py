import pytest

from mqns.entity.memory import QuantumMemory, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import WernerStateEntanglement
from mqns.models.qubit import Qubit
from mqns.simulator import Simulator
from mqns.utils import unwrap


class TwoNodes:
    def __init__(self, *, capacity=1):
        self.n1 = QNode("n1", memory=QuantumMemory("m1", capacity=capacity))
        self.m1 = self.n1.memory

        self.n2 = QNode("n2")
        self.qc = QuantumChannel("qc")
        self.n1.add_qchannel(self.qc)
        self.n2.add_qchannel(self.qc)

        self.s = Simulator(0, 10, accuracy=1000000, install_to=(self.n1, self.n2, self.qc))

    def make_epr(self, name: str) -> WernerStateEntanglement:
        return WernerStateEntanglement(
            name=name,
            decohere_time=self.s.tc + self.m1.t_cohere,
            fidelity_time=self.s.tc,
            src=self.n1,
            dst=self.n2,
        )


def test_write_and_read():
    scenario = TwoNodes(capacity=2)
    mem = scenario.m1

    epr1 = scenario.make_epr("epr1")
    key = "n1_peer_0_0"

    # First allocate memory with path ID
    mem._storage[1][0].key = key

    # Now write with key
    qubit = mem.write(key, epr1)
    assert unwrap(qubit).addr == 1

    # Should fail to write another one in the same slot
    epr2 = scenario.make_epr("epr2")
    with pytest.raises(ValueError, match="contains existing data"):
        mem.write(key, epr2)

    # Should be able to read it
    qubit, data = mem.read(key, has=WernerStateEntanglement, remove=True)
    assert data.name == "epr1"
    assert mem.count == 0

    with pytest.raises(ValueError, match="data at 1 is not"):
        mem.read(qubit.addr, has=WernerStateEntanglement)


def test_decoherence_event():
    scenario = TwoNodes()
    mem = scenario.m1

    epr = scenario.make_epr("epr3")
    qubit = mem.write(0, epr)

    qubit.state = QubitState.ACTIVE
    qubit.state = QubitState.RESERVED
    qubit.state = QubitState.ENTANGLED0

    # Expect it to decohere at t=1.0
    scenario.s.run()

    res, _ = mem.read(0)
    assert res is qubit
    assert qubit.state is QubitState.RELEASE, f"unexpected state {qubit.state}"


def test_sync_qubit():
    scenario = TwoNodes()
    mem = scenario.m1

    q1 = Qubit(name="test_qubit")

    mq = mem.write(0, q1)
    mq.key = "test_qubit"
    assert mem.read("test_qubit") is not None

    assert mem.read("nonexistent") is None
    assert pytest.raises(LookupError, lambda: mem.read("nonexistent", must=True))


def test_sync_qubit_limited():
    scenario = TwoNodes(capacity=5)
    mem = scenario.m1

    for i in range(5):
        q = Qubit(name=f"q{i}")
        mq = mem.write(i, q)
        mq.key = q.name
        assert mem.count == i + 1

    q = Qubit(name="q5")
    with pytest.raises(LookupError, match="qubit q5 not found"):
        mem.write("q5", q)
    assert mem.count == 5

    mq, _ = mem.read("q3", has=Qubit, remove=True)
    assert mem.count == 4
    q = Qubit(name="q6")
    mem.write(mq.addr, q)
    mq.key = q.name
    assert mem.count == 5
    assert mem.read("q6", must=True)[0].addr == 3


def test_remove_empty_qubit_keeps_count():
    scenario = TwoNodes(capacity=2)
    mem = scenario.m1

    epr = scenario.make_epr("epr1")
    mq = mem.write(0, epr)
    mq.key = "n1_n2_0_0"
    assert mem.count == 1

    _, data = mem.read(mq.key, remove=True)
    assert data is epr
    assert mem.count == 0

    # A duplicate release or a late CUTOFF_DISCARD reaches the same qubit again.
    _, data = mem.read(mq.key, remove=True)
    assert data is None
    assert mem.count == 0

    _, data = mem.read(mq.addr, remove=True)
    assert data is None
    assert mem.count == 0
