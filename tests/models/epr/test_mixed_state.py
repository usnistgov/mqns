import pytest

from mqns.models.epr import MixedStateEntanglement
from mqns.models.qubit.const import QUBIT_STATE_0
from mqns.models.qubit.qubit import Qubit
from mqns.simulator import Time


def test_mixed_state(monkeypatch: pytest.MonkeyPatch):
    now = Time(0, accuracy=1000000)
    decohere = now + 5.0

    e1 = MixedStateEntanglement(fidelity=0.95, name="e1", creation_time=now, decoherence_time=decohere)
    e2 = MixedStateEntanglement(fidelity=0.95, name="e2", creation_time=now, decoherence_time=decohere)
    e3 = MixedStateEntanglement.swap(e1, e2, now=now)
    assert e3 is not None
    assert e3.fidelity == pytest.approx(0.903333, abs=1e-6)

    e4 = MixedStateEntanglement(fidelity=0.95, name="e4", creation_time=now, decoherence_time=decohere)
    e5 = MixedStateEntanglement(fidelity=0.95, name="e5", creation_time=now, decoherence_time=decohere)
    e6 = MixedStateEntanglement.swap(e4, e5, now=now)
    assert e6 is not None
    assert e6.fidelity == pytest.approx(0.903333, abs=1e-6)

    monkeypatch.setattr("mqns.models.epr.mixed.get_rand", lambda: 0.0)

    assert e3.purify(e6, now=now) is True
    assert e3.fidelity == pytest.approx(0.929080, abs=1e-6)

    e8 = MixedStateEntanglement(fidelity=0.95, name="e8")
    assert e3.purify(e8, now=now) is True
    assert (e3.a, e3.b, e3.c, e3.d) == pytest.approx((9.183907e-1, 8.179613e-5, 8.179613e-5, 8.144570e-2), rel=1e-6)

    q_in = Qubit(QUBIT_STATE_0)
    q_out = e3.teleportion(q_in)
    assert q_out.state.rho.shape == (2, 2)
    assert q_out.state.rho[0] == pytest.approx([9.998364e-1, 0], rel=1e-6)
    assert q_out.state.rho[1] == pytest.approx([0, 1.635922e-4], rel=1e-6)
