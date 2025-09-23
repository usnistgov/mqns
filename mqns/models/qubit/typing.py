import numpy as np
import numpy.typing as npt

QubitState = npt.NDArray[np.complex128]
"""Single qubit state, shape is (2,1)."""

MultiQubitState = npt.NDArray[np.complex128]
"""Multiple qubit state, shape is (2*n,1)."""

QubitRho = npt.NDArray[np.complex128]
"""Single qubit density matrix, shape is (2,2)."""

MultiQubitRho = npt.NDArray[np.complex128]
"""Multiple qubit density matrix, shape is (2*n,2*n)."""

Operator = npt.NDArray[np.complex128]
"""Operator on multiple qubits, shape is (2*n,2*n)."""

Operator1 = npt.NDArray[np.complex128]
"""Operator on single qubit, shape is (2,2)."""

Operator2 = npt.NDArray[np.complex128]
"""Operator on two qubits, shape is (4,4)."""

Basis = npt.NDArray[np.complex128]
"""Measurement basis, shape is (2,2)."""
