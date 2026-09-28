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

from collections.abc import Sequence
from typing import overload, override

from mqns.models.core import QuantumModel
from mqns.models.core.basis import BASIS_Z, MeasureOutcome
from mqns.models.core.operator import Operator
from mqns.models.core.state import QUBIT_RHO_0, QUBIT_STATE_0, QubitRho, QubitState
from mqns.models.error import DepolarErrorModel, ErrorModel
from mqns.models.error.input import ErrorModelInputBasic, parse_error
from mqns.models.qubit.state import QState


class Qubit(QuantumModel):
    """Qubit within a quantum state."""

    @staticmethod
    def create_multi(
        n: int,
        *,
        rho: QubitRho,
        operate_error: ErrorModelInputBasic = None,
        measure_error: ErrorModelInputBasic = None,
    ) -> list["Qubit"]:
        """
        Create multiple qubits sharing a density matrix.

        Args:
            n: Quantity of qubits.
            rho: Density matrix.
            operate_error: Operate error model.
            measure_error: Measure error model.
        """
        return QState(
            n,
            rho=rho,
            make_qubit=lambda qs: Qubit(
                qs,
                operate_error=operate_error,
                measure_error=measure_error,
            ),
        ).qubits

    name: str
    """Descriptive name."""
    state: QState
    """QState that includes this qubit."""
    operate_error: ErrorModel
    """Operate error model."""
    measure_error: ErrorModel
    """Measure error model."""

    @overload
    def __init__(
        self,
        state: QubitState = QUBIT_STATE_0,
        *,
        operate_error: ErrorModelInputBasic = None,
        measure_error: ErrorModelInputBasic = None,
        name="",
    ):
        """
        Construct with qubit state.

        Args:
            state: Initial state, default is ``|0>``.
            operate_error: Operate error model.
            measure_error: Measure error model.
            name: Descriptive name.
        """

    @overload
    def __init__(
        self,
        *,
        rho: QubitRho,
        operate_error: ErrorModelInputBasic = None,
        measure_error: ErrorModelInputBasic = None,
        name="",
    ):
        """
        Construct with density matrix.

        Args:
            state: Initial density matrix.
            operate_error: Operate error model.
            measure_error: Measure error model.
            name: Descriptive name.
        """

    @overload
    def __init__(
        self,
        state: QState,
        *,
        operate_error: ErrorModelInputBasic,
        measure_error: ErrorModelInputBasic,
    ):
        """
        Used by ``Qubit.create_multi()``.
        """

    def __init__(
        self,
        state: QubitState | QState | None = None,
        *,
        rho: QubitRho = QUBIT_RHO_0,
        operate_error: ErrorModelInputBasic = None,
        measure_error: ErrorModelInputBasic = None,
        name="",
    ):
        self.name = name
        self.state = state if type(state) is QState else QState([self], state=state, rho=rho)
        self.operate_error = parse_error(operate_error, DepolarErrorModel, -1)
        self.measure_error = parse_error(measure_error, DepolarErrorModel, -1)

    def measure(self, basis=BASIS_Z) -> MeasureOutcome:
        """
        Measure this qubit with the specified basis.

        Args:
            basis: Measurement basis, defaults to Z.

        Returns: Measurement outcome 0 or 1.
        """
        self.apply_error(self.measure_error)
        return self.state.measure(self, basis)

    def stochastic_operate(self, operators: Sequence[Operator], probabilities: Sequence[float]) -> None:
        """
        Apply a set of operators with associated probabilities to the qubit.
        It usually turns a pure state into a mixed state.

        Args:
            operators: List of operators, each must operate on a single qubit.
            probabilities: The probability of applying each operator; their sum must be 1.
        """
        i, n = self.state.qubits.index(self), self.state.num
        full_operators = [op.lift(i, n) for op in operators]
        self.state.stochastic_operate(full_operators, probabilities)

    @override
    def apply_error(self, error) -> None:
        error.qubit(self)

    def __repr__(self) -> str:
        return "<qubit " + self.name + ">"
