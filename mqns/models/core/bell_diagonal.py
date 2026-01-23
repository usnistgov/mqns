"""
Mathematical modeling related to Bell-Diagonal state and Pauli channel.
"""

from typing import Literal

import numpy as np

from mqns.models.core.state import ATOL

BellDiagonalProbV = np.ndarray[tuple[Literal[4]], np.dtype[np.float64]]
"""
Bell-Diagonal probability vector.

Elements:
    i: Probability of desired state, i.e. fidelity.
    z: Probability of Z-flip.
    x: Probability of X-flip.
    y: Probability of Y-flip.
"""


def normalize_bell_diagonal_probv(probv: BellDiagonalProbV) -> BellDiagonalProbV:
    """
    Normalize Bell-Diagonal probability vector.
    """
    total = np.sum(probv)
    if total <= ATOL:  # avoid divide-by-zero
        probv.fill(0)
    else:
        probv /= total
    return probv


def make_bell_diagonal_probv(i: float, z: float, x: float, y: float) -> BellDiagonalProbV:
    """
    Construct Bell-Diagonal probability vector.
    """
    return normalize_bell_diagonal_probv(np.array((i, z, x, y), dtype=np.float64))


PauliTransferMat = np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float64]]
"""
Pauli Transfer Matrix (PTM).
"""


def bell_diagonal_probv_to_pauli_transfer_mat(probv: BellDiagonalProbV) -> PauliTransferMat:
    """
    Convert Bell-Diagonal probability vector to Pauli Transfer Matrix.

    * Each row/column corresponds to [I,Z,X,Y].
    * Applying Z swaps I<->Z and X<->Y.
    * Applying X swaps I<->X and Z<->Y.
    * Applying Y swaps I<->Y and Z<->X.
    """
    i, z, x, y = probv
    return np.array(
        [
            [i, z, x, y],
            [z, i, y, x],
            [x, y, i, z],
            [y, x, z, i],
        ],
        dtype=np.float64,
    )
