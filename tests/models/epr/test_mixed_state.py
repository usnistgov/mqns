import pytest

from mqns.models.epr import MixedStateEntanglement
from mqns.models.qubit import Qubit
from mqns.models.qubit.state import QUBIT_STATE_0
from mqns.simulator import Time


def test_fidelity_conversion():
    e = MixedStateEntanglement()
    assert e.fidelity == pytest.approx(1.0, abs=1e-9)
    assert (e.a, e.b, e.c, e.d) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)

    e = MixedStateEntanglement(fidelity=0.7)
    assert e.fidelity == pytest.approx(0.7, abs=1e-9)
    assert (e.a, e.b, e.c, e.d) == pytest.approx((0.7, 0.1, 0.1, 0.1), abs=1e-9)

    e = MixedStateEntanglement(a=5, b=1, c=1, d=1)
    assert e.fidelity == pytest.approx(0.625, abs=1e-9)
    assert (e.a, e.b, e.c, e.d) == pytest.approx((0.625, 0.125, 0.125, 0.125), abs=1e-9)


def test_swap(monkeypatch: pytest.MonkeyPatch):
    now = Time(0, accuracy=1000000)
    decohere = now + 5.0

    e1 = MixedStateEntanglement(fidelity=0.95, name="e1", creation_time=now, decoherence_time=decohere)
    e2 = MixedStateEntanglement(fidelity=0.95, name="e2", creation_time=now, decoherence_time=decohere)
    e3 = MixedStateEntanglement.swap(e1, e2, now=now)
    assert e3 is not None
    assert e3.fidelity == pytest.approx(0.903333, abs=1e-6)


def test_purify_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("mqns.models.epr.mixed.get_rand", lambda: 0.0)
    now = Time(0, accuracy=1000000)
    decohere = now + 5.0

    e3 = MixedStateEntanglement(fidelity=0.9033333333333332, creation_time=now, decoherence_time=decohere)
    e6 = MixedStateEntanglement(fidelity=0.9033333333333332, creation_time=now, decoherence_time=decohere)
    assert e3.purify(e6, now=now) is True
    assert e3.fidelity == pytest.approx(0.929080, abs=1e-6)

    e8 = MixedStateEntanglement(fidelity=0.95, creation_time=now, decoherence_time=decohere)
    assert e3.purify(e8, now=now) is True
    assert (e3.a, e3.b, e3.c, e3.d) == pytest.approx((9.183907e-1, 8.179613e-5, 8.179613e-5, 8.144570e-2), rel=1e-6)


def test_purify_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("mqns.models.epr.mixed.get_rand", lambda: 1.0)
    now = Time(0, accuracy=1000000)
    decohere = now + 5.0

    e3 = MixedStateEntanglement(fidelity=0.9033333333333332, creation_time=now, decoherence_time=decohere)
    e6 = MixedStateEntanglement(fidelity=0.9033333333333332, creation_time=now, decoherence_time=decohere)
    assert e3.purify(e6, now=now) is False


def test_teleportion():
    now = Time(0, accuracy=1000000)
    decohere = now + 5.0

    e3 = MixedStateEntanglement(
        a=0.918390707337142,
        b=8.17961301968751e-05,
        c=8.17961301968751e-05,
        d=0.08144570040246428,
        creation_time=now,
        decoherence_time=decohere,
    )

    q_in = Qubit(QUBIT_STATE_0)
    q_out = e3.teleportation(q_in)
    assert q_out.state.rho.shape == (2, 2)
    assert q_out.state.rho[0] == pytest.approx([9.998364e-1, 0], rel=1e-6)
    assert q_out.state.rho[1] == pytest.approx([0, 1.635922e-4], rel=1e-6)
