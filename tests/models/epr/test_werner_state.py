import numpy as np
import pytest

from mqns.models.epr import WernerStateEntanglement
from mqns.models.qubit.const import QUBIT_STATE_P
from mqns.simulator import Time


def micros(us: int) -> Time:
    """Produce timestamp in microseconds."""
    return Time(us, accuracy=1000000)


def test_fidelity_conversion():
    e = WernerStateEntanglement()
    assert e.fidelity == pytest.approx(1.0, abs=1e-9)

    e = WernerStateEntanglement(fidelity=0.85)
    assert e.fidelity == pytest.approx(0.85, abs=1e-9)


def test_swap_success(monkeypatch: pytest.MonkeyPatch):
    e1 = WernerStateEntanglement(fidelity=0.9, creation_time=micros(1000))
    e2 = WernerStateEntanglement(fidelity=0.8, creation_time=micros(2000))
    e1.decoherence_time = micros(3000)
    e2.decoherence_time = micros(4000)

    monkeypatch.setattr("mqns.models.epr.werner.get_rand", lambda: 0.1)
    ne = e1.swapping(e2, ps=1.0)

    assert ne is not None
    assert ne.fidelity < min(e1.fidelity, e2.fidelity)  # swapping reduces fidelity
    assert ne.creation_time == micros(1000)
    assert ne.decoherence_time == micros(3000)
    assert not ne.is_decoherenced


@pytest.mark.xfail
def test_swap_fidelity():
    """
    Validate fidelity calculation after swaps.
    """
    decoherence_time = micros(1000000)  # 1 second
    decoherence_rate = 1 / decoherence_time.sec

    e1 = WernerStateEntanglement(fidelity=0.99, creation_time=micros(1000))
    e1.decoherence_time = e1.creation_time + decoherence_time
    e2 = WernerStateEntanglement(fidelity=0.99, creation_time=micros(2000))
    e2.decoherence_time = e2.creation_time + decoherence_time
    e3 = WernerStateEntanglement(fidelity=0.99, creation_time=micros(3000))
    e3.decoherence_time = e3.creation_time + decoherence_time

    ne1_time = micros(2500)
    e1.store_error_model((ne1_time - e1.creation_time).sec, decoherence_rate)
    e2.store_error_model((ne1_time - e2.creation_time).sec, decoherence_rate)
    assert e1.w == pytest.approx(0.983711102, abs=1e-6)
    assert e2.w == pytest.approx(0.985680493, abs=1e-6)
    ne1 = e1.swapping(e2, ps=1.0)
    assert ne1 is not None
    assert ne1.w == pytest.approx(0.969624844, abs=1e-6)
    assert ne1.fidelity == pytest.approx(0.977218633, abs=1e-6)

    ne2_time = micros(3500)
    ne1.store_error_model((ne2_time - ne1.creation_time).sec, decoherence_rate)
    e3.store_error_model((ne2_time - e3.creation_time).sec, decoherence_rate)
    assert ne1.w == pytest.approx(0.967687533, abs=1e-6)
    assert e3.w == pytest.approx(0.985680493, abs=1e-6)
    ne2 = ne1.swapping(e3, ps=1.0)
    assert ne2 is not None
    assert ne2.w == pytest.approx(0.953830724, abs=1e-6)
    assert ne2.fidelity == pytest.approx(0.965373043, abs=1e-6)


def test_swap_failure(monkeypatch: pytest.MonkeyPatch):
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


def test_purify_success(monkeypatch: pytest.MonkeyPatch):
    e1 = WernerStateEntanglement(fidelity=0.85)
    e2 = WernerStateEntanglement(fidelity=0.85)

    monkeypatch.setattr("mqns.models.epr.werner.get_rand", lambda: 0.1)
    assert e1.purify(e2) is True

    assert not e1.is_decoherenced
    assert e2.is_decoherenced
    assert e1.fidelity > 0.85


def test_purify_failure(monkeypatch: pytest.MonkeyPatch):
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
    e = WernerStateEntanglement()
    e.store_error_model(t=1.0, decoherence_rate=0.5)
    assert 0 < e.fidelity < 1.0


def test_transfer_error_model():
    e = WernerStateEntanglement()
    e.transfer_error_model(length=10.0, decoherence_rate=0.1)
    assert 0 < e.fidelity < 1.0


def test_to_qubits_non_decohered():
    e = WernerStateEntanglement()
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
