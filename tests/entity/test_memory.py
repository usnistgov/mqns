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
from mqns.models.epr import BaseEntanglement, WernerStateEntanglement
from mqns.models.qubit import Qubit
from mqns.simulator import Simulator


def test_write_and_read_with_path_and_key():
    mem = QuantumMemory("mem", capacity=2, decoherence_rate=1)
    mem.assign(QuantumChannel("qc"), mem.capacity)
    node = QNode("n1")
    node.set_memory(mem)

    sim = Simulator(0, 10)
    node.install(sim)

    epr = WernerStateEntanglement(name="epr1")
    epr.creation_time = sim.tc
    epr.src = node
    epr.dst = QNode("peer")
    key = "n1_peer_0_0"

    # First allocate memory with path ID
    addrs = mem.allocate(0, PathDirection.LEFT, ch_name="qc")
    assert len(addrs) == 1
    addr = addrs[0]
    mem._storage[addr][0].active = key

    # Now write with path_id and key
    result = mem.write(epr, path_id=0, key=key)
    assert result is not None
    assert result.addr == addr

    # Should fail to write another one in the same slot
    epr2 = WernerStateEntanglement(name="epr2")
    epr2.creation_time = sim.tc
    epr2.src = node
    epr2.dst = QNode("peer2")
    assert mem.write(epr2, path_id=0, key=key) is None

    # Should be able to read it
    epr1Read = mem.read("epr1")  # destructive reading
    assert epr1Read is not None
    qubit, data = epr1Read
    assert isinstance(data, BaseEntanglement)
    assert data.name == "epr1"
    assert mem._usage == 0

    assert pytest.raises(ValueError, lambda: mem.read(qubit.addr, must=True))


def test_channel_qubit_assignment_and_search():
    mem = QuantumMemory("mem", capacity=3, decoherence_rate=1)
    node = QNode("n2")
    node.set_memory(mem)

    sim = Simulator(0, 10)
    node.install(sim)

    ch = QuantumChannel("qch", length=10)
    addrs = mem.assign(ch)
    assert len(addrs) == 1

    # Assigned qubit should now be returned by get_channel_qubits
    qubits = mem.get_channel_qubits("qch")
    assert len(qubits) == 1
    q, data = qubits[0]
    assert q.qchannel == ch
    assert data is None


def test_decoherence_event_removes_qubit():
    mem = QuantumMemory("mem", decoherence_rate=1)

    node = QNode("n3")
    node.set_memory(mem)

    sim = Simulator(0, 5)
    node.install(sim)

    epr = WernerStateEntanglement(name="epr3", fidelity=1.0)
    epr.creation_time = sim.tc
    qubit = mem.write(epr)
    assert epr.decoherence_time is not None

    assert qubit is not None
    qubit.state = QubitState.ACTIVE
    qubit.state = QubitState.RESERVED
    qubit.state = QubitState.ENTANGLED0

    # Expect it to decohere at t=1.0
    sim.run()

    res = mem.get("epr3")
    assert res is None
    assert qubit.state == QubitState.RELEASE


def test_memory_clear_and_deallocate():
    mem = QuantumMemory("mem", capacity=2, decoherence_rate=1)
    mem.assign(QuantumChannel("qc"), mem.capacity)
    node = QNode("n4")
    node.set_memory(mem)

    sim = Simulator(0, 5)
    node.install(sim)

    for i in range(2):
        q = WernerStateEntanglement(name=f"epr{i}", fidelity=1.0)
        q.creation_time = sim.tc
        q.src = node
        q.dst = QNode("peer")
        assert mem.write(q)

    assert mem.is_full()
    mem.clear()
    assert not mem.is_full()

    # Test deallocate
    addrs = mem.allocate(7, PathDirection.LEFT, ch_name="qc")
    assert len(addrs) == 1
    addr = addrs[0]
    mem.deallocate(addr)
    with pytest.raises(IndexError):
        mem.deallocate(999)  # invalid


def test_qubit_reservation_behavior():
    mem = QuantumMemory("mem", capacity=2, decoherence_rate=1)
    mem.assign(QuantumChannel("qc"), mem.capacity)
    node = QNode("n5")
    node.set_memory(mem)

    sim = Simulator(0, 5)
    node.install(sim)

    addrs = mem.allocate(42, PathDirection.LEFT, ch_name="qc")
    assert len(addrs) == 1
    addr1 = addrs[0]
    q1 = mem._storage[addr1][0]
    q1.active = "n5_n6_42_" + str(addr1)

    epr = WernerStateEntanglement(name="eprX")
    epr.creation_time = sim.tc
    epr.src = node
    epr.dst = QNode("n6")

    # Must match on both path_id and key
    result = mem.write(epr, path_id=42, key=q1.active)
    assert result is not None
    assert result.addr == addr1


def test_memory_sync_qubit():
    m = QuantumMemory("m1")
    n1 = QNode("n1")
    n1.set_memory(m)
    q1 = Qubit(name="test_qubit")

    s = Simulator(0, 10, 1000)
    n1.install(s)

    assert m.write(q1)
    assert m.read("test_qubit") is not None

    assert m.get("nonexistent") is None
    assert m.read("nonexistent") is None
    assert pytest.raises(IndexError, lambda: m.get("nonexistent", must=True))
    assert pytest.raises(IndexError, lambda: m.read("nonexistent", must=True))


def test_memory_sync_qubit_limited():
    m = QuantumMemory("m1", capacity=5)
    n1 = QNode(name="n1")
    n1.set_memory(m)

    s = Simulator(0, 10, 1000)
    n1.install(s)

    for i in range(5):
        q = Qubit(name="q" + str(i + 1))
        assert m.write(q)
        assert m.count == i + 1

    q = Qubit(name="q5")
    assert not m.write(q)
    assert m.is_full()

    q = m.read("q4")
    assert q is not None
    assert m.count == 4
    assert not m.is_full()
    q = Qubit(name="q6")
    assert m.write(q)
    assert m.is_full()
    assert m.get("q6", must=True)[0].addr == 3


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
    n1.set_memory(m)

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
