#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import overload

import numpy as np

from mqns.models.core import ATOL, BASIS_Z, Basis, MeasureOutcome, Operator, QuantumModel, QubitRho, QubitState
from mqns.models.core.state import QUBIT_STATE_0, check_qubit_rho, qubit_rho_remove, qubit_rho_to_state, qubit_state_to_rho
from mqns.utils import rng


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
        state: QubitState = QUBIT_STATE_0,
        rho: QubitRho | None = None,
    ):
        """
        Args:
            qubits: list of qubits in this state.
            state: state vector, ignored if ``rho`` is specified.
            rho: density matrix.
        """
        self.qubits = qubits
        """List of qubits in this state."""
        self.rho: QubitRho
        """Density matrix."""

        if rho is None:
            self.rho = qubit_state_to_rho(state, self.num)
        else:
            self.rho = check_qubit_rho(rho, self.num)

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

    def operate(self, operator: Operator) -> None:
        """
        Apply an operator to the state.

        Args:
            operator: the operator, which must have the correct dimension.
        """
        self.rho = operator(self.rho)

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

        new_state: QubitRho = np.zeros_like(self.rho)
        for op, p in zip(operators, prob):
            new_state += p * op(self.rho)
        self.rho = check_qubit_rho(new_state, self.num)

    def equal(self, other_state: "QState") -> bool:
        """Compare two state vectors, return True if they are the same

        Args:
            other_state (QState): the second QState

        """
        return np.all(self.rho == other_state.rho).item()

    def state(self) -> QubitState | None:
        """
        Convert to state vector if this is a pure state.

        Returns: Either a state vector, or None if this is a mixed state.
        """
        return qubit_rho_to_state(self.rho, self.num)

    def __repr__(self) -> str:
        return str(self.rho)


class Qubit(QuantumModel):
    """Represent a qubit."""

    @overload
    def __init__(
        self,
        state: QubitState = QUBIT_STATE_0,
        *,
        operate_decoherence_rate=0.0,
        measure_decoherence_rate=0.0,
        name="",
    ):
        """
        Construct with qubit state.

        Args:
            state: initial state, default is ``|0>``.
            operate_decoherence_rate: operate decoherence rate.
            measure_decoherence_rate: measure decoherence rate.
            name: descriptive name.
        """

    @overload
    def __init__(
        self,
        *,
        rho: QubitRho,
        operate_decoherence_rate=0.0,
        measure_decoherence_rate=0.0,
        name="",
    ):
        """
        Construct with density matrix.

        Args:
            state: initial density matrix.
            operate_decoherence_rate: operate decoherence rate.
            measure_decoherence_rate: measure decoherence rate.
            name: descriptive name.
        """

    def __init__(
        self,
        state: QubitState = QUBIT_STATE_0,
        *,
        rho: QubitRho | None = None,
        operate_decoherence_rate=0.0,
        measure_decoherence_rate=0.0,
        name="",
    ):
        self.name = name
        """Descriptive name."""
        self.state = QState([self], state=state, rho=rho)
        """QState that includes this qubit."""
        self.operate_decoherence_rate = operate_decoherence_rate
        """Operate decoherence rate."""
        self.measure_decoherence_rate = measure_decoherence_rate
        """Measure decoherence rate."""

    def measure(self, basis=BASIS_Z) -> MeasureOutcome:
        """
        Measure this qubit with the specified basis.

        Args:
            basis: Measurement basis, defaults to Z.

        Returns: Measurement outcome 0 or 1.
        """
        self.measure_error_model(decoherence_rate=self.measure_decoherence_rate)
        return self.state.measure(self, basis)

    def operate(self, operator: Operator) -> None:
        """Apply an operator on this qubit."""
        self.operate_error_model(self.operate_decoherence_rate)
        full_operator = operator.lift(self.state.qubits.index(self), self.state.num)
        self.state.operate(full_operator)

    def stochastic_operate(self, operators: list[Operator] = [], probabilities: list[float] = []) -> None:
        """
        Apply a set of operators with associated probabilities to the qubit.
        It usually turns a pure state into a mixed state.

        Args:
            operators: a list of operators, each must operate on a single qubit.
            probabilities: the probability of applying each operator; their sum must be 1.
        """
        i, n = self.state.qubits.index(self), self.state.num
        full_operators: list[Operator] = []
        for j, op in enumerate(operators):
            full_operators.append(op.lift(i, n))
        self.state.stochastic_operate(full_operators, probabilities)

    def __repr__(self) -> str:
        if self.name is not None:
            return "<qubit " + self.name + ">"
        return super().__repr__()
