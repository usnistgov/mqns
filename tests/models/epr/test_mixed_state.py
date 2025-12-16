from mqns.models.epr import MixedStateEntanglement
from mqns.models.qubit.const import QUBIT_STATE_0
from mqns.models.qubit.qubit import Qubit
from mqns.simulator import Time


def test_mixed_state():
    now = Time(0, accuracy=1000000)
    decohere = now + 5.0

    e1 = MixedStateEntanglement(fidelity=0.95, name="e1", creation_time=now, decoherence_time=decohere)
    e2 = MixedStateEntanglement(fidelity=0.95, name="e2", creation_time=now, decoherence_time=decohere)
    e3 = MixedStateEntanglement.swap(e1, e2, now=now)
    assert e3 is not None
    print(e3.fidelity)

    e4 = MixedStateEntanglement(fidelity=0.95, name="e4", creation_time=now, decoherence_time=decohere)
    e5 = MixedStateEntanglement(fidelity=0.95, name="e5", creation_time=now, decoherence_time=decohere)
    e6 = MixedStateEntanglement.swap(e4, e5, now=now)
    assert e6 is not None
    print(e6.fidelity)

    e7 = e3.distillation(e6)
    if e7 is None:
        print("distillation failed")
        return
    print(e7.fidelity)

    e8 = MixedStateEntanglement(fidelity=0.95, name="e8")
    e9 = e7.distillation(e8)
    if e9 is None:
        print("distillation failed")
        return
    print(e9.fidelity, e9.b, e9.c, e9.d)

    q_in = Qubit(QUBIT_STATE_0)
    q_out = e9.teleportion(q_in)
    print(q_out.state.rho)


if __name__ == "__main__":
    test_mixed_state()
