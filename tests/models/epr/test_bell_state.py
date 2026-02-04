from mqns.models.core.state import (
    BELL_RHO_PHI_P,
    BELL_STATE_PHI_P,
    QUBIT_STATE_1,
    qubit_rho_classify_noise,
    qubit_state_equal,
)
from mqns.models.epr import BellStateEntanglement
from mqns.models.qubit import Qubit


def test_teleportation():
    c0, c1 = 0, 0
    for _ in range(1000):
        q0 = Qubit(QUBIT_STATE_1)
        e1 = BellStateEntanglement(name="e0")
        q2 = e1.teleportation(q0)
        if q2.measure() == 0:
            c0 += 1
        else:
            c1 += 1
    assert c0 == 0
    assert c1 == 1000


def test_to_qubits():
    e = BellStateEntanglement()

    q0, q1 = e.to_qubits()
    assert e.is_decoherenced

    assert q0.state is q1.state
    assert qubit_rho_classify_noise(BELL_RHO_PHI_P, q0.state.rho) == "IDENTICAL"

    state = q0.state.state()
    assert state is not None  # pure state
    assert qubit_state_equal(BELL_STATE_PHI_P, state)
