import numpy as np

from mqns.models.qubit.operator import OPERATOR_PAULI_X, OPERATOR_PAULI_Y, OPERATOR_PAULI_Z, Operator
from mqns.models.qubit.state import (
    QUBIT_STATE_0,
    QUBIT_STATE_1,
    QUBIT_STATE_L,
    QUBIT_STATE_N,
    QUBIT_STATE_P,
    QUBIT_STATE_R,
    QubitState,
)


class Basis:
    """Measurement basis."""

    def __init__(
        self,
        name: str,
        *,
        observable: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        s0: QubitState,
        s1: QubitState,
    ):
        self.name = name
        """Measurement basis name."""
        self.observable = observable
        """Pauli observable."""
        self.s0 = s0
        """Outcome 0."""
        self.s1 = s1
        """Outcome 1."""
        self.m0 = Operator(s0 @ s0.conj().T, check_unitary=False)
        """Projection operator 0."""
        self.m1 = Operator(s1 @ s1.conj().T, check_unitary=False)
        """Projection operator 1."""

    def __repr__(self) -> str:
        return f"<Basis {self.name}>"


BASIS_Z = Basis(
    "Z",
    observable=OPERATOR_PAULI_Z.u,
    s0=QUBIT_STATE_0,
    s1=QUBIT_STATE_1,
)
"""Measurement basis Z: projects onto ``|0>`` and ``|1>``."""

BASIS_X = Basis(
    "X",
    observable=OPERATOR_PAULI_X.u,
    s0=QUBIT_STATE_P,
    s1=QUBIT_STATE_N,
)
"""Measurement basis X: projects onto ``|+>`` and ``|->``.."""

BASIS_Y = Basis(
    "Y",
    observable=OPERATOR_PAULI_Y.u,
    s0=QUBIT_STATE_R,
    s1=QUBIT_STATE_L,
)
"""Measurement basis Y: projects onto ``|R>`` and ``|L>``.."""

BASIS_BY_NAME: dict[str, Basis] = {
    "Z": BASIS_Z,
    "X": BASIS_X,
    "Y": BASIS_Y,
}
