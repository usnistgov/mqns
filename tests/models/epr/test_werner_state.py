import pytest

from mqns.models.core.state import (
    BELL_RHO_PHI_P,
    BELL_STATE_PHI_P,
    QUBIT_STATE_P,
    qubit_rho_classify_noise,
    qubit_state_equal,
)
from mqns.models.epr import Entanglement, WernerStateEntanglement
from mqns.models.error import DephaseErrorModel
from mqns.simulator import Time
from mqns.utils import rng


def micros(us: int) -> Time:
    """Produce timestamp in microseconds."""
    return Time(us, accuracy=1000000)


def test_fidelity_conversion():
    e = WernerStateEntanglement()
    assert e.fidelity == pytest.approx(1.0, abs=1e-9)

    e = WernerStateEntanglement(fidelity=0.85)
    assert e.fidelity == pytest.approx(0.85, abs=1e-9)


def test_swap_success(monkeypatch: pytest.MonkeyPatch):
    e1 = WernerStateEntanglement(fidelity=0.9, creation_time=micros(1000), decoherence_time=micros(3000))
    e2 = WernerStateEntanglement(fidelity=0.8, creation_time=micros(2000), decoherence_time=micros(4000))

    e1.read, e2.read = True, True
    monkeypatch.setattr(rng, "random", lambda: 0.1)
    ne = Entanglement.swap(e1, e2, now=micros(2500))

    assert ne is not None
    assert ne.fidelity < min(e1.fidelity, e2.fidelity)  # swapping reduces fidelity
    assert ne.creation_time == micros(2500)
    assert ne.decoherence_time == micros(3000)
    assert not ne.is_decoherenced


def test_swap_fidelity():
    """
    Validate fidelity calculation after swaps.
    """
    mem_dt = micros(1000000)  # memory decoherence time: 1 second
    dephase_error = DephaseErrorModel().set(t=0, rate=2 / mem_dt.time_slot)

    def dephase(e: Entanglement, now: Time):
        assert not e.read
        dephase_error.set(t=(now - e.creation_time).time_slot)
        e.apply_error(dephase_error)
        e.read = True

    e1t = micros(1000)
    e2t = micros(2000)
    e3t = micros(3000)
    e1 = WernerStateEntanglement(fidelity=0.99, creation_time=e1t, decoherence_time=e1t + mem_dt)
    e2 = WernerStateEntanglement(fidelity=0.99, creation_time=e2t, decoherence_time=e2t + mem_dt)
    e3 = WernerStateEntanglement(fidelity=0.99, creation_time=e3t, decoherence_time=e3t + mem_dt)

    ne1t = micros(2500)
    dephase(e1, ne1t)
    dephase(e2, ne1t)
    ne1 = Entanglement.swap(e1, e2, now=ne1t)
    assert e1.w == pytest.approx(0.983711102, abs=1e-6)
    assert e2.w == pytest.approx(0.985680493, abs=1e-6)
    assert ne1 is not None
    assert ne1.w == pytest.approx(0.969624844, abs=1e-6)
    assert ne1.fidelity == pytest.approx(0.977218633, abs=1e-6)

    ne2t = micros(3500)
    dephase(ne1, ne2t)
    dephase(e3, ne2t)
    ne2 = Entanglement.swap(ne1, e3, now=ne2t)
    assert ne1.w == pytest.approx(0.967687533, abs=1e-6)
    assert e3.w == pytest.approx(0.985680493, abs=1e-6)
    assert ne2 is not None
    assert ne2.w == pytest.approx(0.953830724, abs=1e-6)
    assert ne2.fidelity == pytest.approx(0.965373043, abs=1e-6)


def test_swap_failure(monkeypatch: pytest.MonkeyPatch):
    now = micros(0)
    decohere = now + 1.0
    e1 = WernerStateEntanglement(fidelity=0.9, creation_time=now, decoherence_time=decohere)
    e2 = WernerStateEntanglement(fidelity=0.8, creation_time=now, decoherence_time=decohere)

    monkeypatch.setattr(rng, "random", lambda: 0.99)
    ne = Entanglement.swap(e1, e2, now=now, ps=0.5)

    assert ne is None
    assert e1.is_decoherenced
    assert e2.is_decoherenced


def test_swap_decohered_inputs():
    now = micros(0)
    decohere = now + 1.0
    e1 = WernerStateEntanglement(fidelity=0.9, creation_time=now, decoherence_time=decohere)
    e2 = WernerStateEntanglement(fidelity=0.8, creation_time=now, decoherence_time=decohere)
    e1.is_decoherenced = True

    assert Entanglement.swap(e1, e2, now=now) is None


def test_purify_success(monkeypatch: pytest.MonkeyPatch):
    now = Time(0, accuracy=1000000)
    e1 = WernerStateEntanglement(fidelity=0.85)
    e2 = WernerStateEntanglement(fidelity=0.85)

    monkeypatch.setattr(rng, "random", lambda: 0.1)
    assert e1.purify(e2, now=now) is True

    assert not e1.is_decoherenced
    assert e2.is_decoherenced
    assert e1.fidelity > 0.85


def test_purify_failure(monkeypatch: pytest.MonkeyPatch):
    now = Time(0, accuracy=1000000)
    e1 = WernerStateEntanglement(fidelity=0.5)
    e2 = WernerStateEntanglement(fidelity=0.5)

    monkeypatch.setattr(rng, "random", lambda: 0.99)
    assert e1.purify(e2, now=now) is False

    assert e1.is_decoherenced
    assert e2.is_decoherenced
    assert e1.fidelity == 0


def test_purify_decohered_input():
    now = Time(0, accuracy=1000000)
    e1 = WernerStateEntanglement(fidelity=0.85)
    e2 = WernerStateEntanglement(fidelity=0.85)
    e1.is_decoherenced = True

    assert e1.purify(e2, now=now) is False
    assert e1.is_decoherenced
    assert e1.fidelity == 0


def test_to_qubits_maximal():
    e = WernerStateEntanglement()
    q0, q1 = e.to_qubits()
    assert e.is_decoherenced

    assert q0.state is q1.state
    print(q0.state)
    assert qubit_rho_classify_noise(BELL_RHO_PHI_P, q0.state.rho) == 0

    state = q0.state.state()
    assert state is not None  # pure state
    assert qubit_state_equal(BELL_STATE_PHI_P, state)

    v0 = q0.measure()
    v1 = q1.measure()
    assert v0 == v1


def test_to_qubits_mixed():
    e = WernerStateEntanglement(fidelity=0.9)
    q0, q1 = e.to_qubits()
    assert e.is_decoherenced

    assert q0.state is q1.state
    print(q0.state)
    assert qubit_rho_classify_noise(BELL_RHO_PHI_P, q0.state.rho) == 2
    assert q0.state.state() is None  # mixed state


def test_to_qubits_decohered():
    e = WernerStateEntanglement()
    e.is_decoherenced = True
    q0, q1 = e.to_qubits()
    assert e.is_decoherenced

    assert q0.state is not q1.state  # disjoint state
    for q in q0, q1:
        state = q.state.state()
        assert state is not None
        assert qubit_state_equal(QUBIT_STATE_P, state)
