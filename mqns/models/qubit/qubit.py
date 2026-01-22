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

from typing import Any, Literal, cast

import numpy as np

from mqns.models.core import QuantumModel
from mqns.models.core.basis import BASIS_BY_NAME
from mqns.models.core.operator import Operator
from mqns.models.core.state import (
    ATOL,
    QUBIT_STATE_0,
    QubitRho,
    QubitState,
    check_qubit_rho,
    qubit_rho_remove,
    qubit_rho_to_state,
    qubit_state_to_rho,
)
from mqns.models.qubit.gate import SingleQubitGate
from mqns.utils import rng


class QState:
    """QState represents the state of one or multiple qubits."""

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
        name: str | None = None,
    ):
        """Args:
        qubits (List[Qubit]): a list of qubits in this quantum state
        state: the state vector of this state, either ``state`` or ``rho`` can be used to present a state
        rho: the density matrix of this state, either ``state`` or ``rho`` can be used to present a state;
             if both ``state`` and ``rho`` are specified, ``rho`` takes priority
        name (str): the name of this state

        """
        self.num = len(qubits)
        self.name = name
        self.qubits = qubits
        self.rho: QubitRho

        if rho is None:
            self.rho = qubit_state_to_rho(state, self.num)
        else:
            self.rho = check_qubit_rho(rho, self.num)

    def measure(self, qubit: "Qubit", base: Literal["Z", "X", "Y"] = "Z") -> int:
        """
        Measure this qubit using the specified basis.

        Args:
            qubit: the qubit to be measured.
            base: the measure base, "Z", "X" or "Y".

        Returns: Measurement outcome 0 or 1.
        """
        basis = BASIS_BY_NAME[base]

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
        self.num -= 1
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

        new_state: QubitRho = np.zeros((2**self.num, 2**self.num), dtype=np.complex128)
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
        """If the state is a pure state, return the state vector, or return None

        Returns:
            The pure state vector

        """
        return qubit_rho_to_state(self.rho, self.num)

    def __repr__(self) -> str:
        if self.name is not None:
            return "<qubit state " + self.name + ">"
        return str(self.rho)


class Qubit(QuantumModel):
    """Represent a qubit"""

    def __init__(
        self,
        state: QubitState = QUBIT_STATE_0,
        rho: QubitRho | None = None,
        operate_decoherence_rate: float = 0,
        measure_decoherence_rate: float = 0,
        name: str | None = None,
    ):
        """Args:
        state (list): the initial state of a qubit, default is |0> = [1, 0]^T
        operate_decoherence_rate (float): the operate decoherence rate
        measure_decoherence_rate (float): the measure decoherence rate
        name (str): the qubit's name

        """
        self.name = name
        self.state = QState([self], state=state, rho=rho)
        self.operate_decoherence_rate = operate_decoherence_rate
        self.measure_decoherence_rate = measure_decoherence_rate

    def measure(self):
        """Measure this qubit using Z basis

        Returns:
            0: QUBIT_STATE_0 state
            1: QUBIT_STATE_1 state

        """
        self.measure_error_model(decoherence_rate=self.measure_decoherence_rate)
        return self.state.measure(self)

    def measureX(self):
        """Measure this qubit using X basis.

        Returns:
            0: QUBIT_STATE_P state
            1: QUBIT_STATE_N state

        """
        self.measure_error_model(self.measure_decoherence_rate)
        return self.state.measure(self, "X")

    def measureY(self):
        """Measure this qubit using Y basis.
        Only for not entangled qubits.

        Returns:
            0: QUBIT_STATE_R state
            1: QUBIT_STATE_L state

        """
        self.measure_error_model(self.measure_decoherence_rate)
        return self.state.measure(self, "Y")

    def measureZ(self):
        """Measure this qubit using Z basis

        Returns:
            0: QUBIT_STATE_0 state
            1: QUBIT_STATE_1 state

        """
        self.measure_error_model(self.measure_decoherence_rate)
        return self.measure()

    def operate(self, operator: SingleQubitGate | Operator) -> None:
        """Apply an operator on this qubit, with operator error model."""
        self.operate_error_model(self.operate_decoherence_rate)
        self._operate_without_error(operator)

    def _operate_without_error(self, operator: SingleQubitGate | Operator) -> None:
        """Apply an operator on this qubit, without operator error model."""
        if isinstance(operator, SingleQubitGate):
            operator(self)
            return
        full_operator = operator.lift(self.state.qubits.index(self), self.state.num)
        self.state.operate(full_operator)

    def stochastic_operate(self, operators: list[SingleQubitGate | Operator] = [], probabilities: list[float] = []) -> None:
        """
        Apply a set of operators with associated probabilities to the qubit.
        It usually turns a pure state into a mixed state.

        Args:
            operators: a list of operators, each must operator on a single qubit.
            probabilities: the probability of applying each operator; their sum must be 1.
        """
        i, n = self.state.qubits.index(self), self.state.num
        full_operators: list[Operator] = []
        for j, op_gate in enumerate(operators):
            op = cast(Any, op_gate)._operator if isinstance(op_gate, SingleQubitGate) else op_gate
            assert type(op) is Operator, f"operators[{j}] does not contain Operator matrix"
            full_operators.append(op.lift(i, n))
        self.state.stochastic_operate(full_operators, probabilities)

    def __repr__(self) -> str:
        if self.name is not None:
            return "<qubit " + self.name + ">"
        return super().__repr__()
