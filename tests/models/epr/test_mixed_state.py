import pytest

from mqns.models.epr import MixedStateEntanglement
from mqns.models.qubit import Qubit
from mqns.models.qubit.state import (
    BELL_RHO_PHI_N,
    BELL_RHO_PHI_P,
    BELL_RHO_PSI_N,
    BELL_RHO_PSI_P,
    BELL_STATE_PHI_N,
    BELL_STATE_PHI_P,
    BELL_STATE_PSI_N,
    BELL_STATE_PSI_P,
    QUBIT_STATE_0,
    QubitRho,
    QubitState,
    qubit_rho_classify_noise,
    qubit_state_are_equal,
)
from mqns.simulator import Time
from mqns.utils import rng


def test_fidelity_conversion():
    e = MixedStateEntanglement()
    assert e.fidelity == pytest.approx(1.0, abs=1e-9)
    assert (e.i, e.z, e.x, e.y) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)

    e = MixedStateEntanglement(fidelity=0.7)
    assert e.fidelity == pytest.approx(0.7, abs=1e-9)
    assert (e.i, e.z, e.x, e.y) == pytest.approx((0.7, 0.1, 0.1, 0.1), abs=1e-9)

    e = MixedStateEntanglement(i=4, z=2, x=1, y=1)
    assert e.fidelity == pytest.approx(0.5, abs=1e-9)
    assert (e.i, e.z, e.x, e.y) == pytest.approx((0.5, 0.25, 0.125, 0.125), abs=1e-9)


def test_swap():
    now = Time(0, accuracy=1000000)
    decohere = now + 5.0

    e1 = MixedStateEntanglement(fidelity=0.95, name="e1", creation_time=now, decoherence_time=decohere)
    e2 = MixedStateEntanglement(fidelity=0.95, name="e2", creation_time=now, decoherence_time=decohere)
    e3 = MixedStateEntanglement.swap(e1, e2, now=now)
    assert e3 is not None
    assert e3.fidelity == pytest.approx(0.903333, abs=1e-6)


def test_purify_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rng, "random", lambda: 0.0)
    now = Time(0, accuracy=1000000)
    decohere = now + 5.0

    e3 = MixedStateEntanglement(fidelity=0.9033333333333332, creation_time=now, decoherence_time=decohere)
    e6 = MixedStateEntanglement(fidelity=0.9033333333333332, creation_time=now, decoherence_time=decohere)
    assert e3.purify(e6, now=now) is True
    assert e3.fidelity == pytest.approx(0.929080, abs=1e-6)

    e8 = MixedStateEntanglement(fidelity=0.95, creation_time=now, decoherence_time=decohere)
    assert e3.purify(e8, now=now) is True
    assert (e3.i, e3.z, e3.x, e3.y) == pytest.approx((9.183907e-1, 8.179613e-5, 8.179613e-5, 8.144570e-2), rel=1e-6)


def test_purify_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rng, "random", lambda: 1.0)
    now = Time(0, accuracy=1000000)
    decohere = now + 5.0

    e3 = MixedStateEntanglement(fidelity=0.9033333333333332, creation_time=now, decoherence_time=decohere)
    e6 = MixedStateEntanglement(fidelity=0.9033333333333332, creation_time=now, decoherence_time=decohere)
    assert e3.purify(e6, now=now) is False


def test_teleportion():
    now = Time(0, accuracy=1000000)
    decohere = now + 5.0

    e3 = MixedStateEntanglement(
        i=0.918390707337142,
        z=0.08144570040246428,
        x=8.17961301968751e-05,
        y=8.17961301968751e-05,
        creation_time=now,
        decoherence_time=decohere,
    )

    q_in = Qubit(QUBIT_STATE_0)
    q_out = e3.teleportation(q_in)
    assert q_out.state.rho.shape == (2, 2)
    assert q_out.state.rho[0] == pytest.approx([9.998364e-1, 0], rel=1e-6)
    assert q_out.state.rho[1] == pytest.approx([0, 1.635922e-4], rel=1e-6)


@pytest.mark.parametrize(
    ("i", "z", "x", "y", "state", "rho", "measured_same"),
    [
        (1, 0, 0, 0, BELL_STATE_PHI_P, BELL_RHO_PHI_P, True),
        (0, 1, 0, 0, BELL_STATE_PHI_N, BELL_RHO_PHI_N, True),
        (0, 0, 1, 0, BELL_STATE_PSI_P, BELL_RHO_PSI_P, False),
        (0, 0, 0, 1, BELL_STATE_PSI_N, BELL_RHO_PSI_N, False),
    ],
)
def test_to_qubits_maximal(i: float, z: float, x: float, y: float, state: QubitState, rho: QubitRho, measured_same: bool):
    e = MixedStateEntanglement(i=i, z=z, x=x, y=y)

    qlist = e.to_qubits()
    assert e.is_decoherenced
    assert len(qlist) == 2

    q0, q1 = qlist
    assert q0.state is q1.state
    assert qubit_rho_classify_noise(rho, q0.state.rho) == 0

    pure_state = q0.state.state()
    assert pure_state is not None  # pure state
    assert qubit_state_are_equal(state, pure_state)

    v0 = q0.measure()
    v1 = q1.measure()
    if measured_same:
        assert v0 == v1
    else:
        assert v0 != v1


def test_to_qubits_dephase():
    e = MixedStateEntanglement()
    e.dephase(1.0, 1 / 5.0)
    print(e.i, e.z, e.x, e.y)

    qlist = e.to_qubits()
    assert e.is_decoherenced
    assert len(qlist) == 2

    q0, q1 = qlist
    assert q0.state is q1.state
    print(q0.state)
    assert qubit_rho_classify_noise(BELL_RHO_PHI_P, q0.state.rho) == 1
    assert q0.state.state() is None  # mixed state


def test_to_qubits_depolarize():
    e = MixedStateEntanglement()
    e.depolarize(1.0, 1 / 5.0)
    print(e.i, e.z, e.x, e.y)

    qlist = e.to_qubits()
    assert e.is_decoherenced
    assert len(qlist) == 2

    q0, q1 = qlist
    assert q0.state is q1.state
    print(q0.state)
    assert qubit_rho_classify_noise(BELL_RHO_PHI_P, q0.state.rho) == 2
    assert q0.state.state() is None  # mixed state
