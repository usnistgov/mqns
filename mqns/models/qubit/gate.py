import functools
from collections.abc import Callable

import numpy as np

from mqns.models.core.operator import (
    OPERATOR_H,
    OPERATOR_PAULI_I,
    OPERATOR_PAULI_X,
    OPERATOR_PAULI_Y,
    OPERATOR_PAULI_Z,
    OPERATOR_PHASE_SHIFT,
    OPERATOR_RX,
    OPERATOR_RY,
    OPERATOR_RZ,
    OPERATOR_S,
    OPERATOR_T,
    Operator,
)
from mqns.models.qubit.qubit import Qubit
from mqns.models.qubit.state import QState

_id = np.identity(2, dtype=np.complex128)
_p0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)  # projector matrix |0><0|
_p1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)  # projector matrix |1><1|


def operate_single(qubit: Qubit, op: Operator) -> None:
    """
    Apply a single-qubit operator.
    """
    state = qubit.state
    i, n = state.qubits.index(qubit), state.num
    full_op = op.lift(i, n)

    qubit.apply_error(qubit.operate_error)
    state.operate(full_op)


def operate_controlled(q0: Qubit, q1: Qubit, op: Operator) -> None:
    """
    Apply a controlled operator.

    Args:
        q0: controller qubit.
        q1: target qubit.
        op: single-qubit operator.
    """
    assert op.n == 1
    state = QState.joint(q0, q1)
    n = state.num
    i0, i1 = state.qubits.index(q0), state.qubits.index(q1)
    assert i0 != i1, "Qubits must be distinct"

    mats0 = [_id] * n
    mats1 = [_id] * n
    mats0[i0] = _p0
    mats1[i0] = _p1
    mats1[i1] = op.u

    full_op = functools.reduce(np.kron, mats0) + functools.reduce(np.kron, mats1)

    for q in q0, q1:
        q.apply_error(q.operate_error)
    state.operate(full_op)


def operate_cc(q0: Qubit, q1: Qubit, q2: Qubit, op: Operator) -> None:
    """
    Apply a controlled-controlled operator.

    Args:
        q0: first controller qubit.
        q1: second controller qubit.
        q2: target qubit.
        op: single-qubit operator.
    """
    assert op.n == 1
    QState.joint(q0, q1)
    state = QState.joint(q0, q2)
    n = state.num
    i0, i1, i2 = state.qubits.index(q0), state.qubits.index(q1), state.qubits.index(q2)
    assert len({i0, i1, i2}) == 3, "Qubits must be distinct"

    mats_zz = [_id] * n
    mats_zz[i0] = _p1
    mats_zz[i1] = _p1

    mats_11 = mats_zz.copy()
    mats_11[i2] = op.u

    # full_id         = (|00><00| + |01><01| + |10><10| + |11><11|)⊗I
    # p11_on_controls = (|11><11|)⊗I
    # u11_on_target   = (|11><11|)⊗U
    # full_op         = (|00><00| + |01><01| + |10><10|)⊗I + (|11><11|)⊗U
    full_id = np.identity(2**n, dtype=np.complex128)
    p11_on_controls = functools.reduce(np.kron, mats_zz)
    u11_on_target = functools.reduce(np.kron, mats_11)
    full_op = full_id - p11_on_controls + u11_on_target

    for q in q0, q1, q2:
        q.apply_error(q.operate_error)
    state.operate(full_op)


def _make_single(op: Operator):
    def f(qubit: Qubit) -> None:
        operate_single(qubit, op)

    return f


def _make_rotate(op: Callable[[float], Operator]):
    def f(qubit: Qubit, theta=np.pi / 4) -> None:
        operate_single(qubit, op(theta))

    return f


def _make_controlled(op: Operator):
    def f(q0: Qubit, q1: Qubit) -> None:
        return operate_controlled(q0, q1, op)

    return f


def _make_cc(op: Operator):
    def f(q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        return operate_cc(q0, q1, q2, op)

    return f


I = _make_single(OPERATOR_PAULI_I)
"""Pauli I Gate"""
X = _make_single(OPERATOR_PAULI_X)
"""Pauli X Gate"""
Y = _make_single(OPERATOR_PAULI_Y)
"""Pauli Y Gate"""
Z = _make_single(OPERATOR_PAULI_Z)
"""Pauli Z Gate"""
H = _make_single(OPERATOR_H)
"""Hadamard Gate"""
T = _make_single(OPERATOR_T)
"""T gate (pi/4 shift gate)"""
S = _make_single(OPERATOR_S)
"""S gate (pi/2 shift gate)"""


R = _make_rotate(OPERATOR_PHASE_SHIFT)
"""R gate (phase shift gate)"""
RX = _make_rotate(OPERATOR_RX)
"""Rx gate (X rotate gate)"""
RY = _make_rotate(OPERATOR_RY)
"""Ry gate (Y rotate gate)"""
RZ = _make_rotate(OPERATOR_RZ)
"""Rz gate (Z rotate gate)"""


def U(qubit: Qubit, op: Operator) -> None:
    """Arbitrary single qubit operation gate"""
    return operate_single(qubit, op)


CX = _make_controlled(OPERATOR_PAULI_X)
"""Controlled Pauli-X gate"""
CY = _make_controlled(OPERATOR_PAULI_Y)
"""Controlled Pauli-Y gate"""
CZ = _make_controlled(OPERATOR_PAULI_Z)
"""Controlled Pauli-Z gate"""
CNOT = CX
"""Controlled NOT gate"""


def _make_controlled_rotate(op: Callable[[float], Operator]):
    def f(q0: Qubit, q1: Qubit, theta=np.pi / 4) -> None:
        operate_controlled(q0, q1, op(theta))

    return f


CR = _make_controlled_rotate(OPERATOR_PHASE_SHIFT)
"""Controlled Phase Rotate Gate"""


def Swap(q0: Qubit, q1: Qubit) -> None:
    """Swap gate: swap the states of two qubits."""
    state = QState.joint(q0, q1)
    i0 = state.qubits.index(q0)
    i1 = state.qubits.index(q1)
    assert i0 != i1, "Qubits must be distinct"
    state.qubits[i0], state.qubits[i1] = state.qubits[i1], state.qubits[i0]


Toffoli = _make_cc(OPERATOR_PAULI_X)
"""Toffoli Gate"""
