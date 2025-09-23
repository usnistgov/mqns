import numpy as np

from mqns.models.epr import WernerStateEntanglement
from mqns.models.qubit.const import QUBIT_STATE_P
from mqns.simulator import Time


def test_fidelity_conversion():
    e = WernerStateEntanglement(fidelity=0.85)
    assert e.fidelity == 0.85


def test_swap_success(monkeypatch):
    e1 = WernerStateEntanglement(fidelity=0.9)
    e2 = WernerStateEntanglement(fidelity=0.8)
    e1.creation_time = Time(time_slot=1, accuracy=1000)
    e2.creation_time = Time(time_slot=2, accuracy=1000)
    e1.decoherence_time = Time(time_slot=3, accuracy=1000)
    e2.decoherence_time = Time(time_slot=4, accuracy=1000)

    monkeypatch.setattr("mqns.models.epr.werner.get_rand", lambda: 0.1)

    ne = e1.swapping(e2, ps=1.0)
    assert ne is not None
    assert ne.fidelity < min(e1.fidelity, e2.fidelity)  # swapping reduces fidelity
    assert ne.creation_time == Time(time_slot=1, accuracy=1000)
    assert ne.decoherence_time == Time(time_slot=3, accuracy=1000)
    assert not ne.is_decoherenced


def test_swap_failure(monkeypatch):
    e1 = WernerStateEntanglement(fidelity=0.9)
    e2 = WernerStateEntanglement(fidelity=0.8)
    monkeypatch.setattr("mqns.models.epr.werner.get_rand", lambda: 0.99)

    ne = e1.swapping(e2, ps=0.5)
    assert ne is None
    assert e1.is_decoherenced
    assert e2.is_decoherenced


def test_swap_decohered_inputs():
    e1 = WernerStateEntanglement(fidelity=0.9)
    e2 = WernerStateEntanglement(fidelity=0.8)
    e1.is_decoherenced = True

    assert e1.swapping(e2) is None


def test_purify_success(monkeypatch):
    e1 = WernerStateEntanglement(fidelity=0.85)
    e2 = WernerStateEntanglement(fidelity=0.85)
    monkeypatch.setattr("mqns.models.epr.werner.get_rand", lambda: 0.1)

    assert e1.purify(e2) is True
    assert not e1.is_decoherenced
    assert e2.is_decoherenced
    assert e1.fidelity > 0.85


def test_purify_failure(monkeypatch):
    e1 = WernerStateEntanglement(fidelity=0.5)
    e2 = WernerStateEntanglement(fidelity=0.5)
    monkeypatch.setattr("mqns.models.epr.werner.get_rand", lambda: 0.99)

    assert e1.purify(e2) is False
    assert e1.is_decoherenced
    assert e2.is_decoherenced
    assert e1.fidelity == 0


def test_purify_decohered_input():
    e1 = WernerStateEntanglement(fidelity=0.85)
    e2 = WernerStateEntanglement(fidelity=0.85)
    e1.is_decoherenced = True

    assert e1.purify(e2) is False
    assert e1.is_decoherenced
    assert e1.fidelity == 0


def test_store_error_model():
    e = WernerStateEntanglement(fidelity=1.0)
    e.store_error_model(t=1.0, decoherence_rate=0.5)
    assert 0 < e.fidelity < 1.0


def test_transfer_error_model():
    e = WernerStateEntanglement(fidelity=1.0)
    e.transfer_error_model(length=10.0, decoherence_rate=0.1)
    assert 0 < e.fidelity < 1.0


def test_to_qubits_non_decohered():
    e = WernerStateEntanglement(fidelity=1.0)
    qlist = e.to_qubits()
    assert len(qlist) == 2
    assert qlist[0].state is qlist[1].state
    assert e.is_decoherenced


def test_to_qubits_after_decoherence():
    e = WernerStateEntanglement(fidelity=0.8)
    e.is_decoherenced = True
    qlist = e.to_qubits()

    assert len(qlist) == 2

    for q in qlist:
        state_vector = q.state.state()
        assert state_vector is not None
        assert np.allclose(state_vector, QUBIT_STATE_P, rtol=0, atol=1e-15)
