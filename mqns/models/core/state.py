"""
Definitions and constants for qubit state vector and density matrix.
"""

from collections.abc import Iterable
from typing import Literal

import numpy as np

ATOL = 1e-9
"""Absolute numerical tolerance for floating-point calculations."""

type QubitState = np.ndarray[tuple[int, Literal[1]], np.dtype[np.complex128]]
"""Qubit state vector for N qubits, shape is (2**N, 1)."""


def check_qubit_state(state: np.ndarray, n=1) -> QubitState:
    """
    Validate that ``state`` is a state vector for ``n`` qubits.

    Args:
        state: NDArray.
        n: expected number of qubits.

    Raises:
        AssertionError - ``state`` has wrong shape or is not normalized.

    Returns: Validated input.
    """
    # Complex Amplitudes: the entries of the vector must be complex numbers.
    assert np.iscomplexobj(state)
    # Dimensionality: the vector must reside in a Hilbert space of dimension 2**n.
    assert state.shape == (2**n, 1)
    # Normalization: the total probability must sum to 1.
    assert np.isclose(np.linalg.norm(state), 1.0, atol=ATOL)
    return state


def build_qubit_state(input: Iterable[complex], n=1) -> QubitState:
    """
    Build a state vector for ``n`` qubits.

    Args:
        input: list of 2**n probabilities.
        n: number of qubits.

    Returns: Normalized qubit state.
    """
    array = np.array(input, dtype=np.complex128)
    column = array[:, np.newaxis]
    norm = np.linalg.norm(column)
    column /= norm
    return check_qubit_state(column, n)


def qubit_state_normalize_phase(state: QubitState) -> QubitState:
    """
    Standardize the global phase of a state vector.
    """
    # Find element(s) with significant magnitude.
    found, _ = np.where(np.abs(state) > ATOL)
    if len(found) < 1:
        return state

    # Extract the phase of the first found element.
    value = state[found[0]]
    phase = np.conj(value) / np.abs(value)

    # Multiply the vector by that phase.
    return state * phase


def qubit_state_equal(s0: QubitState, s1: QubitState) -> bool:
    """Compare qubit state vectors for equality."""
    return np.allclose(s0, s1, atol=ATOL)


type QubitRho = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
"""Qubit density matrix for N qubits, shape is (2**N, 2**N)."""


def _check_rho_shape(rho: np.ndarray, n: int) -> None:
    assert np.iscomplexobj(rho)
    # Dimensionality: the matrix must be a square with dimensions (2**n,2**n).
    assert rho.shape == (2**n, 2**n)


def check_qubit_rho(rho: np.ndarray, n=1, *, maybe_zero=False) -> QubitRho:
    """
    Validate that ``rho`` is a density matrix for ``n`` qubits.

    Args:
        rho: NDArray.
        n: expected number of qubits.
        maybe_zero: if True, don't error if the matrix is all zeros.

    Raises:
        AssertionError - ``rho`` has wrong shape or is not normalized.

    Returns: Validated input.
    """
    _check_rho_shape(rho, n)
    # Unit Trace: the sum of the diagonal elements must be exactly 1,
    # representing a total probability of 100%.
    trace = np.trace(rho)
    if maybe_zero and np.abs(trace) < ATOL:
        assert np.all(rho < ATOL)
    else:
        assert np.isclose(trace, 1.0, atol=ATOL)
    # Hermiticity: the matrix must equal its own conjugate transpose.
    assert np.allclose(rho, rho.conj().T, atol=ATOL)
    # Positive Semi-Definiteness: all eigenvalues must be non-negative
    # This ensures no state has a negative probability of occurring.
    assert np.all(np.linalg.eigvalsh(rho) >= -ATOL)
    return rho


def normalize_qubit_rho(rho: np.ndarray, n=1, *, maybe_zero=False) -> QubitRho:
    """
    Normal ``rho`` to a density matrix for ``n`` qubits.

    Args:
        rho: NDArray.
        n: expected number of qubits.
        maybe_zero: if True, don't error if the matrix is all zeros.

    Raises:
        AssertionError - ``rho`` has wrong shape.

    Returns: Validated input.
    """
    _check_rho_shape(rho, n)

    # Force hermiticity
    rho = (rho + rho.conj().T) / 2

    # Force positive semi-definiteness (spectral clipping)
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    if np.any(eigenvalues < 0):
        eigenvalues = np.maximum(eigenvalues, 0)
        rho = (eigenvectors * eigenvalues) @ eigenvectors.conj().T  # V @ diag(λ) @ V_dagger

    # Force unit trace
    trace = np.real(np.trace(rho))
    if maybe_zero and np.abs(trace) < ATOL:
        rho = np.zeros_like(rho, dtype=np.complex128)
    else:
        rho /= trace

    return rho


def qubit_state_to_rho(state: QubitState, n=1) -> QubitRho:
    """Convert qubit state vector to density matrix."""
    return check_qubit_rho(np.outer(state, np.conj(state)), n)


def qubit_rho_to_state(rho: QubitRho, n=1) -> QubitState | None:
    """
    Convert density matrix to state vector.

    Args:
        rho: density matrix.
        n: number of qubits.

    Returns:
        * State vector if ``rho`` represents a pure state.
        * None if ``rho`` represents a mixed state.
    """
    purity = np.linalg.norm(rho) ** 2
    if not np.isclose(purity, 1.0, atol=ATOL):
        return None

    # eigh returns eigenvalues in ascending order.
    # For a pure state, the largest eigenvalue should be 1.0, located at the end.
    _, eigenvectors = np.linalg.eigh(rho)
    psi = eigenvectors[:, -1]
    psi = psi.reshape((-1, 1))

    # Global phase correction: first non-zero element is real
    phase = np.exp(-1j * np.angle(psi[np.argmax(np.abs(psi) > ATOL)]))
    psi *= phase
    return check_qubit_state(psi, n)


def qubit_rho_equal(rho0: QubitRho, rho1: QubitRho) -> bool:
    """Compare density matrices for equality."""
    return np.allclose(rho0, rho1, atol=ATOL)


def qubit_rho_classify_noise(ideal: QubitRho, noisy: QubitRho) -> int:
    """
    Identify what kind of noise has been applied to transform ``ideal`` state to ``noisy`` state.

    Args:
        * 0 - identical.
        * 1 - pure dephasing noise.
        * 2 - depolarizing, bit-flip, or amplitude damping noise.
    """
    if qubit_rho_equal(ideal, noisy):
        return 0
    if np.allclose(ideal.diagonal(), noisy.diagonal(), atol=ATOL):
        return 1
    return 2


def qubit_rho_remove(rho: QubitRho, i: int, n: int) -> QubitRho:
    """
    Remove the i-th qubit from a density matrix of n qubits.

    Args:
        rho: a density matrix of n qubits.
        i: the index of the qubit to be removed.
        n: total number of qubits before the operation.

    Returns:
        Density matrix of n-1 qubits.
    """
    res = rho.reshape((2, 2) * n)
    res: np.ndarray = res.trace(axis1=i, axis2=n + i)
    dim = 2 ** (n - 1)
    res = res.reshape((dim, dim))
    return normalize_qubit_rho(res, n - 1, maybe_zero=True)


QUBIT_STATE_0 = build_qubit_state((1, 0))
"""Single qubit state: ``|0>``."""
QUBIT_STATE_1 = build_qubit_state((0, 1))
"""Single qubit state: ``|1>``."""
QUBIT_STATE_P = build_qubit_state((1, 1))
"""Single qubit state: ``|+>``."""
QUBIT_STATE_N = build_qubit_state((1, -1))
"""Single qubit state: ``|->``."""
QUBIT_STATE_R = build_qubit_state((1, 1j))
"""Single qubit state: ``|R>``."""
QUBIT_STATE_L = build_qubit_state((1, -1j))
"""Single qubit state: ``|L>``."""

QUBIT_RHO_0 = qubit_state_to_rho(QUBIT_STATE_0)
"""Density matrix of ``QUBIT_STATE_0``."""
QUBIT_RHO_1 = qubit_state_to_rho(QUBIT_STATE_1)
"""Density matrix of ``QUBIT_STATE_1``."""
QUBIT_RHO_P = qubit_state_to_rho(QUBIT_STATE_P)
"""Density matrix of ``QUBIT_STATE_P``."""
QUBIT_RHO_N = qubit_state_to_rho(QUBIT_STATE_N)
"""Density matrix of ``QUBIT_STATE_N``."""
QUBIT_RHO_R = qubit_state_to_rho(QUBIT_STATE_R)
"""Density matrix of ``QUBIT_STATE_R``."""
QUBIT_RHO_L = qubit_state_to_rho(QUBIT_STATE_L)
"""Density matrix of ``QUBIT_STATE_L``."""

BELL_STATE_PHI_P = build_qubit_state((1, 0, 0, 1), 2)
"""Two-qubit maximally entangled Bell state: ``|Φ+>`` i.e. ``(|00>+|11>)/sqrt(2)``."""
BELL_STATE_PHI_N = build_qubit_state((1, 0, 0, -1), 2)
"""Two-qubit maximally entangled Bell state: ``|Φ->`` i.e. ``(|00>-|11>)/sqrt(2)``."""
BELL_STATE_PSI_P = build_qubit_state((0, 1, 1, 0), 2)
"""Two-qubit maximally entangled Bell state: ``|Ψ+>`` i.e. ``(|01>+|10>)/sqrt(2)``."""
BELL_STATE_PSI_N = build_qubit_state((0, 1, -1, 0), 2)
"""Two-qubit maximally entangled Bell state: ``|Ψ->`` i.e. ``(|01>-|10>)/sqrt(2)``."""

BELL_RHO_PHI_P = qubit_state_to_rho(BELL_STATE_PHI_P, 2)
"""Density matrix of ``BELL_STATE_PHI_P``."""
BELL_RHO_PHI_N = qubit_state_to_rho(BELL_STATE_PHI_N, 2)
"""Density matrix of ``BELL_STATE_PHI_N``."""
BELL_RHO_PSI_P = qubit_state_to_rho(BELL_STATE_PSI_P, 2)
"""Density matrix of ``BELL_STATE_PSI_P``."""
BELL_RHO_PSI_N = qubit_state_to_rho(BELL_STATE_PSI_N, 2)
"""Density matrix of ``BELL_STATE_PSI_N``."""
