from collections.abc import Iterable

import numpy as np
import pytest

from mqns.models.core.operator import OPERATOR_RX, OPERATOR_RY
from mqns.models.core.state import (
    QUBIT_STATE_0,
    QUBIT_STATE_1,
    build_qubit_state,
    qubit_rho_equal,
    qubit_state_equal,
    qubit_state_normalize_phase,
)
from mqns.models.qubit import Qubit
from mqns.models.qubit.gate import CNOT, CR, CZ, RX, RY, H, Swap, Toffoli, U, X, Z

_sqrt1_2 = 1 / np.sqrt(2)


def make_qubits(*values: int) -> tuple[Qubit, ...]:
    return tuple(
        Qubit(
            state=QUBIT_STATE_1 if v == 1 else QUBIT_STATE_0,
            name=f"q{i}",
        )
        for i, v in enumerate(values)
    )


def check_state(qubits: list[Qubit], expected_input: Iterable[complex]):
    state = qubits[0].state
    assert state.qubits == qubits
    pure = state.state()
    assert pure is not None
    pure = qubit_state_normalize_phase(pure)
    expected = build_qubit_state(expected_input, len(qubits))
    assert qubit_state_equal(pure, expected)


def test_bell_state():
    # Build Psi+ state
    q0, q1 = make_qubits(0, 0)
    H(q0)
    CNOT(q0, q1)

    # They should share a state
    assert q0.state is q1.state

    # Measuring a qubit would unshare the state
    c0 = q0.measure()
    assert q0.state is not q1.state
    c1 = q1.measure()

    # Measurement outcome should be the same
    assert c0 == c1


def test_rx():
    q0, q1 = make_qubits(0, 0)
    RX(q0)
    U(q1, OPERATOR_RX(theta=np.pi / 4))
    assert qubit_rho_equal(q0.state.rho, q1.state.rho)


def test_ry():
    q0, q1 = make_qubits(0, 0)
    RY(q0)
    U(q1, OPERATOR_RY(theta=np.pi / 4))
    assert qubit_rho_equal(q0.state.rho, q1.state.rho)


@pytest.mark.parametrize(
    ("v1", "expected"),
    [
        (0, [1, 0, 1, 0]),
        (1, [0, 1, 0, -1]),
    ],
)
def test_cz(v1: int, expected: Iterable[complex]):
    q0, q1 = make_qubits(0, v1)
    H(q0)
    CZ(q0, q1)
    check_state([q0, q1], expected)


@pytest.mark.parametrize(
    ("v1", "theta", "expected"),
    [
        (0, np.pi / 4, [1, 0, 1, 0]),
        (1, np.pi / 4, [0, _sqrt1_2, 0, _sqrt1_2 * np.exp(1j * np.pi / 4)]),
        (0, np.pi / 2, [1, 0, 1, 0]),
        (1, np.pi / 2, [0, _sqrt1_2, 0, _sqrt1_2 * np.exp(1j * np.pi / 2)]),
    ],
)
def test_cr(v1: int, theta: float, expected: Iterable[complex]):
    q0, q1 = make_qubits(0, v1)
    H(q0)
    CR(q0, q1, theta=theta)
    check_state([q0, q1], expected)


def test_swap():
    q0, q1, q2 = make_qubits(0, 1, 0)

    # q0,q1,q2=0,1,0
    Swap(q0, q1)
    # q0,q1,q2=1,0,0
    Swap(q0, q2)
    # q0,q1,q2=0,0,1

    assert q0.measure() == 0
    assert q1.measure() == 0
    assert q2.measure() == 1


@pytest.mark.parametrize(
    ("v0", "v1", "expected_q2"),
    [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 1),
    ],
)
def test_toffoli(v0: int, v1: int, expected_q2: int):
    q0, q1, q2 = make_qubits(v0, v1, 0)
    Toffoli(q0, q1, q2)
    assert q2.measure() == expected_q2


def test_bsm():
    q0, q1, q2, q3 = make_qubits(0, 0, 0, 0)

    H(q0)
    CNOT(q0, q1)

    H(q2)
    CNOT(q2, q3)

    CNOT(q1, q2)
    H(q1)
    c0 = q2.measure()
    c1 = q1.measure()

    if c0 == 1 and c1 == 0:
        X(q3)
    elif c0 == 0 and c1 == 1:
        Z(q3)
    elif c0 == 1 and c1 == 1:
        X(q3)
        Z(q3)
    assert q0.measure() == q3.measure()
