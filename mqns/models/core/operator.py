"""
Definitions and constants for qubit operator.
"""

import functools
from typing import Final, cast, final

import numpy as np

from mqns.models.core.state import ATOL, QubitRho, QubitState


@final
class Operator:
    n: Final[int]
    """Number of qubits this operator can be used with."""

    u: Final[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]
    """Operator matrix."""

    u_dagger: Final[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]
    """Hermitian conjugate of the operator matrix."""

    def __init__(self, input: np.ndarray | list[list[complex]], n=1, *, check_unitary=True):
        """
        Build an operator for ``n`` qubits.

        Args:
            input: Matrix with ``(2**n, 2**n)`` dimension.
            n: Number of qubits.
            check_unitary: If True, enforce the operator is unitary.
        """
        self.n = n
        self.u = np.array(input, dtype=np.complex128)
        self.u.flags.writeable = False
        self.u_dagger = self.u.conj().T
        self.u_dagger.flags.writeable = False
        self._validate(check_unitary)

    def _validate(self, check_unitary: bool) -> None:
        dim = 2**self.n
        assert self.u.shape == (dim, dim), f"Expected ({dim}, {dim}), got {self.u.shape}"
        if check_unitary:
            is_unitary = np.allclose(self.u_dagger @ self.u, np.identity(dim, dtype=np.complex128), atol=ATOL)
            assert is_unitary, "Operator is not unitary (U^dagger @ U != I)"

    def __call__[T: (QubitState | QubitRho)](self, state: T) -> T:
        """
        Apply an operator on qubits.

        Args:
            state: Either a state vector or a density matrix for ``self.n`` qubits.

        Raises:
            ValueError: Mismatch between operator size and state size.

        Returns:
            Transformed state vector or density matrix.
        """
        if state.shape[0] != self.u.shape[0]:
            raise ValueError("state dimension does not match operator dimension")

        if state.shape[1] == 1:  # state vector
            return cast(T, self.u @ state)
        else:  # density matrix
            return cast(T, self.u @ state @ self.u_dagger)

    def lift(self, i: int, n: int, *, check_unitary=True) -> "Operator":
        """
        Expand a single-qubit operator to apply on the i-th qubit of a n-qubit state.

        Args:
            self: Single-qubit operator.
            i: Target qubit index.
            n: Number of qubits in the state vector or density matrix.

        Raises:
            AssertionError: This is not a single-qubit operator.
            ValueError: ``i`` or ``n`` is out of range.

        Returns:
            An n-qubit operator.
        """
        assert self.n == 1, "can only lift 1-qubit operator"
        if not (0 <= i < n):
            raise ValueError("i or n is out of range")
        if n == 1:
            return self

        # full_matrix = I⊗..⊗U⊗..⊗I where the i-th matrix is U
        mats = [OPERATOR_PAULI_I.u] * n
        mats[i] = self.u
        full_matrix = functools.reduce(np.kron, mats)
        return Operator(full_matrix, n, check_unitary=check_unitary)


def OPERATOR_RX(theta: float):
    """
    Build an operator for rotation around the X-axis: ``exp(-i*theta*X/2)``.

    Args:
        theta: Bloch Sphere rotation angle, in radians.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return Operator([[c, -1j * s], [-1j * s, c]])


def OPERATOR_RY(theta: float):
    """
    Build an operator for rotation around the Y-axis: ``exp(-i*theta*Y/2)``.

    Args:
        theta: Bloch Sphere rotation angle, in radians.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return Operator([[c, -s], [s, c]])


def OPERATOR_RZ(theta: float):
    """
    Build an operator for rotation around the Z-axis: ``exp(-i*theta*Z/2)``.

    Args:
        theta: Bloch Sphere rotation angle, in radians.
    """
    return Operator([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]])


def OPERATOR_PHASE_SHIFT(theta: float):
    """
    Build an operator for phase shift gate: ``|1> -> exp(i*theta)|1>``.

    Args:
        theta: Relative phase, in radians.
    """
    return Operator([[1, 0], [0, np.exp(1j * theta)]])


_sqrt1_2: float = 1 / np.sqrt(2)
OPERATOR_H = Operator([[_sqrt1_2, _sqrt1_2], [_sqrt1_2, -_sqrt1_2]])
"""Hadamard operator."""

OPERATOR_S = OPERATOR_PHASE_SHIFT(np.pi / 2)
"""Phase shift operator."""
OPERATOR_T = OPERATOR_PHASE_SHIFT(np.pi / 4)
"""T operator."""

OPERATOR_PAULI_I = Operator(np.identity(2))
"""Pauli identity operator."""
OPERATOR_PAULI_Z = Operator([[1, 0], [0, -1]])
"""Pauli-Z operator."""
OPERATOR_PAULI_X = Operator([[0, 1], [1, 0]])
"""Pauli-X operator."""
OPERATOR_PAULI_Y = Operator([[0, -1j], [1j, 0]])
"""Pauli-Y operator."""

OPERATOR_CNOT = Operator([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], n=2)
"""Controlled Not operator -- flip the second qubit if the first is ``|1>``."""
OPERATOR_SWAP = Operator([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], n=2)
"""SWAP operator."""
