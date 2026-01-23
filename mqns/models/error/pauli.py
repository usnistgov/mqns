import functools
from collections.abc import Iterable
from typing import Literal, cast, override

import numpy as np

from mqns.models.core.bell_diagonal import bell_diagonal_probv_to_pauli_transfer_mat, make_bell_diagonal_probv
from mqns.models.core.operator import OPERATOR_PAULI_I, OPERATOR_PAULI_X, OPERATOR_PAULI_Y, OPERATOR_PAULI_Z, Operator
from mqns.models.core.state import ATOL
from mqns.models.error.error import ErrorModel

OP_IZXY = [OPERATOR_PAULI_I, OPERATOR_PAULI_Z, OPERATOR_PAULI_X, OPERATOR_PAULI_Y]
_probv0 = make_bell_diagonal_probv(1, 0, 0, 0)


class PauliErrorModel(ErrorModel):
    """
    Pauli error model: one of Z,X,Y gates may be randomly applied with ``p_error`` total probability.
    """

    def __init__(self, name="pauli", *, z=0.0, x=0.0, y=0.0):
        """
        Constructor.

        Args:
            name: name of this error model.
            z: ratio of Z-gate application within ``p_error``.
            x: ratio of X-gate application within ``p_error``.
            y: ratio of Y-gate application within ``p_error``.
        """
        super().__init__(name)

        self.ratios: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]] = np.array([z, x, y], dtype=np.float64)
        """Ratio of Z,X,Y gate application, sum is 1."""
        total = np.sum(self.ratios)
        if total > ATOL:
            self.ratios /= total
        else:
            self.ratios.fill(1 / 3)

        self.probv = _probv0
        """Probability of I,Z,X,Y result."""
        # initial p_survival is 1.0, corresponding to probv=[1,0,0,0]

    @override
    def _prepare(self) -> None:
        z, x, y = self.ratios * self.p_error
        self.probv = make_bell_diagonal_probv(self.p_survival, z, x, y)

        try:
            del self._ptm
        except AttributeError:
            pass

        try:
            del self._stochastic_ops
        except AttributeError:
            pass

    @functools.cached_property
    def _ptm(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float64]]:
        """
        Construct the transition matrix for Bell-diagonal states.
        """
        return bell_diagonal_probv_to_pauli_transfer_mat(self.probv)

    @functools.cached_property
    def _stochastic_ops(self) -> tuple[list[Operator], list[float]]:
        """
        Construct the operators and probabilities for single qubit stochastic operations.
        """
        ops: list[Operator] = []
        prob: list[float] = []
        for o, p in zip(OP_IZXY, cast(Iterable[float], self.probv), strict=True):
            if p > 0:
                ops.append(o)
                prob.append(p)
        assert len(ops) > 0
        return ops, prob

    @override
    def qubit(self, q) -> None:
        ops, prob = self._stochastic_ops
        q.stochastic_operate(ops, prob)

    @override
    def werner(self, q) -> None:
        q.w *= self.p_survival

    @override
    def mixed(self, q) -> None:
        q.set_probv(self._ptm @ q.probv)


class DepolarErrorModel(PauliErrorModel):
    """
    Depolarizing error model: one of Z,X,Y gates may be randomly applied with ``p_error`` probability.
    """

    def __init__(self, name="depolarizing"):
        super().__init__(name, z=1, x=1, y=1)


class DephaseErrorModel(PauliErrorModel):
    """
    Dephasing error model: Z gate may be randomly applied with ``p_error`` probability.
    """

    def __init__(self, name="dephasing"):
        super().__init__(name, z=1)


class BitFlipErrorModel(PauliErrorModel):
    """
    Bit flip error model: X gate may be randomly applied with ``p_error`` probability.
    """

    def __init__(self, name="bit-flip"):
        super().__init__(name, x=1)
