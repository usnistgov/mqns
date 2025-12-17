import pytest

from mqns.entity.memory import (
    MemoryReadRequestEvent,
    MemoryReadResponseEvent,
    MemoryWriteRequestEvent,
    MemoryWriteResponseEvent,
    PathDirection,
    QuantumMemory,
    QubitState,
)
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import WernerStateEntanglement
from mqns.models.qubit import Qubit
from mqns.simulator import Simulator


class TwoNodes:
    def __init__(self, *, capacity=1):
        self.n1 = QNode("n1")
        self.m1 = QuantumMemory("m1", capacity=capacity)
        self.n1.memory = self.m1

        self.n2 = QNode("n2")
        self.qc = QuantumChannel("qc")
        self.n1.add_qchannel(self.qc)
        self.n2.add_qchannel(self.qc)

        self.s = Simulator(0, 10, accuracy=1000000)
        self.n1.install(self.s)
        self.n2.install(self.s)
        self.qc.install(self.s)

    def make_epr(self, name: str) -> WernerStateEntanglement:
        return WernerStateEntanglement(
            name=name,
            creation_time=self.s.tc,
            decoherence_time=self.s.tc + self.m1.decoherence_delay,
            src=self.n1,
            dst=self.n2,
        )


def test_write_and_read_with_path_and_key():
    scenario = TwoNodes(capacity=2)
    mem = scenario.m1
    mem.assign(scenario.qc, n=2)

    epr1 = scenario.make_epr("epr1")
    key = "n1_peer_0_0"

    # First allocate memory with path ID
    addrs = mem.allocate(scenario.qc, 0, PathDirection.L)
    assert len(addrs) == 1
    addr = addrs[0]
    mem._storage[addr][0].active = key

    # Now write with path_id and key
    qubit = mem.write(key, epr1)
    assert qubit is not None
    assert qubit.addr == addr

    # Should fail to write another one in the same slot
    epr2 = scenario.make_epr("epr2")
    with pytest.raises(ValueError, match="contains existing data"):
        mem.write(key, epr2)

    # Should be able to read it
    qubit, data = mem.read("epr1", has=WernerStateEntanglement, remove=True)
    assert data.name == "epr1"
    assert mem._usage == 0

    with pytest.raises(ValueError, match="data at 0 is not"):
        mem.read(qubit.addr, has=WernerStateEntanglement)


def test_channel_qubit_assignment_and_search():
    scenario = TwoNodes(capacity=3)
    mem = scenario.m1

    assigned = mem.assign(scenario.qc, n=2)
    assert len(assigned) == 2

    allocated = mem.allocate(scenario.qc, 7, PathDirection.R, n="all")
    assert len(allocated) == 2

    with pytest.raises(OverflowError, match="insufficient qubits"):
        mem.allocate(scenario.qc, 9, PathDirection.L)

    # Assigned qubit should now be returned by find()
    qubits = list(mem.find(lambda *_: True, qchannel=scenario.qc))
    assert len(qubits) == 2
    for qubit, data in qubits:
        assert qubit.qchannel == scenario.qc
        assert qubit.path_id == 7
        assert qubit.path_direction == PathDirection.R
        assert data is None


def test_decoherence_event_removes_qubit():
    scenario = TwoNodes()
    mem = scenario.m1

    epr = scenario.make_epr("epr3")
    qubit = mem.write(None, epr)

    qubit.state = QubitState.ACTIVE
    qubit.state = QubitState.RESERVED
    qubit.state = QubitState.ENTANGLED0

    # Expect it to decohere at t=1.0
    scenario.s.run()

    res = mem.read("epr3")
    assert res is None
    assert qubit.state == QubitState.RELEASE


def test_memory_clear_and_deallocate():
    scenario = TwoNodes(capacity=2)
    mem = scenario.m1
    mem.assign(scenario.qc, n=2)

    for i in range(2):
        epr = scenario.make_epr(f"epr{i}")
        mem.write(None, epr)

    assert mem.count == 2
    mem.clear()
    assert mem.count == 0

    # Test deallocate
    addrs = mem.allocate(scenario.qc, 7, PathDirection.L)
    assert len(addrs) == 1
    addr = addrs[0]
    mem.deallocate(addr)
    with pytest.raises(IndexError):
        mem.deallocate(999)  # invalid


def test_qubit_reservation_behavior():
    scenario = TwoNodes(capacity=2)
    mem = scenario.m1
    mem.assign(scenario.qc, n=2)

    addrs = mem.allocate(scenario.qc, 42, PathDirection.L)
    assert len(addrs) == 1
    addr1 = addrs[0]
    q1 = mem._storage[addr1][0]
    q1.active = "n5_n6_42_" + str(addr1)

    epr = scenario.make_epr("epr1")

    qubit = mem.write(q1.active, epr)
    assert qubit.addr == addr1


def test_memory_sync_qubit():
    scenario = TwoNodes()
    mem = scenario.m1

    q1 = Qubit(name="test_qubit")

    mem.write(None, q1)
    assert mem.read("test_qubit") is not None

    assert mem.read("nonexistent") is None
    assert pytest.raises(IndexError, lambda: mem.read("nonexistent", must=True))


def test_memory_sync_qubit_limited():
    scenario = TwoNodes(capacity=5)
    mem = scenario.m1

    for i in range(5):
        q = Qubit(name="q" + str(i + 1))
        mem.write(None, q)
        assert mem.count == i + 1

    q = Qubit(name="q5")
    with pytest.raises(IndexError, match="qubit not found"):
        mem.write(None, q)
    assert mem.count == 5

    q = mem.read("q4", remove=True)
    assert q is not None
    assert mem.count == 4
    q = Qubit(name="q6")
    mem.write(None, q)
    assert mem.count == 5
    assert mem.read("q6", must=True)[0].addr == 3


def test_memory_async_qubit():
    class MemoryReadResponseApp(Application):
        def __init__(self):
            super().__init__()
            self.add_handler(self.handleMemoryRead, MemoryReadResponseEvent)
            self.add_handler(self.handleMemoryWrite, MemoryWriteResponseEvent)
            self.nReads = 0
            self.nWrites = 0

        def handleMemoryRead(self, event: MemoryReadResponseEvent) -> bool | None:
            self.nReads += 1
            result = event.result

            print("self.simulator.tc.sec: {}".format(self.simulator.tc))
            print("result: {}".format(result))
            assert self.simulator.tc.sec == pytest.approx(1.5)
            assert result is not None

            qubit, data = result
            assert qubit.addr == 0
            assert isinstance(data, Qubit)

        def handleMemoryWrite(self, event: MemoryWriteResponseEvent) -> bool | None:
            self.nWrites += 1
            result = event.result

            print("self.simulator.tc.sec: {}".format(self.simulator.tc))
            print("result: {}".format(result))
            assert self.simulator.tc.sec == pytest.approx(0.5)
            assert result is not None

            assert result.addr == 0

    n1 = QNode("n1")
    app = MemoryReadResponseApp()
    n1.add_apps(app)

    m = QuantumMemory("m1", delay=0.5)
    n1.memory = m

    s = Simulator(0, 10, 1000)
    n1.install(s)

    q1 = Qubit(name="q1")
    write_request = MemoryWriteRequestEvent(memory=m, qubit=q1, t=s.time(sec=0), by=n1)
    read_request = MemoryReadRequestEvent(memory=m, key="q1", t=s.time(sec=1), by=n1)
    s.add_event(write_request)
    s.add_event(read_request)
    s.run()

    assert app.nReads == 1
    assert app.nWrites == 1
