from mqns.models.epr import BellStateEntanglement
from mqns.models.qubit.const import QUBIT_STATE_1
from mqns.models.qubit.qubit import Qubit


def test_bell_state_epr():
    c0, c1 = 0, 0
    for _ in range(1000):
        q0 = Qubit(QUBIT_STATE_1)
        e1 = BellStateEntanglement(fidelity=1, name="e0")
        q2 = e1.teleportion(q0)
        if q2.measure() == 0:
            c0 += 1
        else:
            c1 += 1
    assert c0 == 0
    assert c1 == 1000


if __name__ == "__main__":
    test_bell_state_epr()
