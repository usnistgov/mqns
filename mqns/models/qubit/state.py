from typing import TYPE_CHECKING

import numpy as np

from mqns.models.core import ATOL, Basis, MeasureOutcome, Operator, QubitRho, QubitState
from mqns.models.core.state import QUBIT_RHO_0, check_qubit_rho, qubit_rho_remove, qubit_rho_to_state, qubit_state_to_rho
from mqns.utils import rng

if TYPE_CHECKING:
    from mqns.models.qubit.qubit import Qubit


class QState:
    """QState tracks the state of one or more qubits."""

    @staticmethod
    def joint(q0: "Qubit", q1: "Qubit") -> "QState":
        """
        Merge QState if necessary so that two qubits share the same QState.
        """
        if q0.state is q1.state:
            return q0.state
        assert set(q0.state.qubits).isdisjoint(q1.state.qubits)
        rho: np.ndarray = np.kron(q0.state.rho, q1.state.rho)
        nq = QState(q0.state.qubits + q1.state.qubits, rho=rho)
        for q in nq.qubits:
            q.state = nq
        return nq

    def __init__(
        self,
        qubits: list["Qubit"],
        *,
        state: QubitState | None = None,
        rho=QUBIT_RHO_0,
    ):
        """
        Args:
            qubits: list of qubits in this state.
            state: state vector.
            rho: density matrix, ignored if ``state`` is specified.
        """
        self.qubits = qubits
        """List of qubits in this state."""
        self.rho = check_qubit_rho(rho, self.num) if state is None else qubit_state_to_rho(state, self.num)
        """Density matrix."""

    @property
    def num(self) -> int:
        """Return number of qubits in this state."""
        return len(self.qubits)

    def measure(self, qubit: "Qubit", basis: Basis) -> MeasureOutcome:
        """
        Measure a qubit using the specified basis.

        Args:
            qubit: the qubit to be measured, which will be removed from the state.
            basis: measurement basis.

        Returns: Measurement outcome 0 or 1.
        """
        try:
            idx = self.qubits.index(qubit)
        except AssertionError:
            raise RuntimeError("qubit not in state")

        # Calculate probability with Born rule
        full_m0 = basis.m0.lift(idx, self.num, check_unitary=False)
        prob_0 = np.real(np.trace(full_m0.u_dagger @ full_m0.u @ self.rho))
        prob_0 = np.clip(prob_0, 0.0, 1.0)  # avoid out-of-range due to floating-point calculation

        # Assign outcome and perform state collapse
        if rng.random() < prob_0:
            ret = 0
            ret_s = basis.s0
            op = full_m0
        else:
            ret = 1
            ret_s = basis.s1
            op = basis.m1.lift(idx, self.num, check_unitary=False)

        # Perform state collapse
        collapsed = op(self.rho)
        collapsed = collapsed / (np.trace(collapsed) or 1.0)

        # Perform partial trace to delete measured qubit
        self.rho = qubit_rho_remove(collapsed, idx, self.num)
        self.qubits.remove(qubit)

        ns = QState([qubit], state=ret_s)
        qubit.state = ns
        return ret

    def operate(self, op: Operator | np.ndarray) -> None:
        """
        Apply an operator to the state.

        Args:
            op: the operator or its matrix with the correct dimension.
        """
        if not isinstance(op, Operator):
            op = Operator(op, self.num)
        self.rho = op(self.rho)

    def stochastic_operate(self, operators: list[Operator] = [], probabilities: list[float] = []) -> None:
        """
        Apply a set of operators with associated probabilities to the state.
        It usually turns a pure state into a mixed state.

        Args:
            operators: a list of operators, each must have the correct dimension.
            probabilities: the probability of applying each operator; their sum must be 1.
        """
        assert len(operators) == len(probabilities), "must have same number of operators and probabilities"
        prob = np.array(probabilities, dtype=np.float64)
        assert np.all(prob >= 0), "each probability must be between 0 and 1"
        assert np.all(prob <= 1), "each probability must be between 0 and 1"
        assert np.isclose(np.sum(prob), 1.0, atol=ATOL), "sum of probabilities must be 1"

        new_rho: QubitRho = np.zeros_like(self.rho)
        for op, p in zip(operators, prob):
            new_rho += p * op(self.rho)
        self.rho = check_qubit_rho(new_rho, self.num)

    def state(self) -> QubitState | None:
        """
        Convert to state vector if this is a pure state.

        Returns: Either a state vector, or None if this is a mixed state.
        """
        return qubit_rho_to_state(self.rho, self.num)

    def __repr__(self) -> str:
        return str(self.rho)
