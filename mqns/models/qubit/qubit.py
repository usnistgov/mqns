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

from typing import overload, override

from mqns.models.core import QuantumModel
from mqns.models.core.basis import BASIS_Z, MeasureOutcome
from mqns.models.core.operator import Operator
from mqns.models.core.state import (
    QUBIT_RHO_0,
    QUBIT_STATE_0,
    QubitRho,
    QubitState,
)
from mqns.models.error import DepolarErrorModel, ErrorModelInput, parse_error
from mqns.models.qubit.state import QState


class Qubit(QuantumModel):
    """Represent a qubit."""

    @overload
    def __init__(
        self,
        state: QubitState = QUBIT_STATE_0,
        *,
        operate_error: ErrorModelInput = None,
        measure_error: ErrorModelInput = None,
        name="",
    ):
        """
        Construct with qubit state.

        Args:
            state: initial state, default is ``|0>``.
            operate_error: operate error model.
            measure_error: measure error model.
            name: descriptive name.
        """

    @overload
    def __init__(
        self,
        *,
        rho: QubitRho,
        operate_error: ErrorModelInput = None,
        measure_error: ErrorModelInput = None,
        name="",
    ):
        """
        Construct with density matrix.

        Args:
            state: initial density matrix.
            operate_error: operate error model.
            measure_error: measure error model.
            name: descriptive name.
        """

    def __init__(
        self,
        state: QubitState | None = None,
        *,
        rho: QubitRho = QUBIT_RHO_0,
        operate_error: ErrorModelInput = None,
        measure_error: ErrorModelInput = None,
        name="",
    ):
        self.name = name
        """Descriptive name."""
        self.state = QState([self], state=state, rho=rho)
        """QState that includes this qubit."""
        self.operate_error = parse_error(operate_error, DepolarErrorModel)
        """Operate error model."""
        self.measure_error = parse_error(measure_error, DepolarErrorModel)
        """Measure error model."""

    def measure(self, basis=BASIS_Z) -> MeasureOutcome:
        """
        Measure this qubit with the specified basis.

        Args:
            basis: Measurement basis, defaults to Z.

        Returns: Measurement outcome 0 or 1.
        """
        self.apply_error(self.measure_error)
        return self.state.measure(self, basis)

    def stochastic_operate(self, operators: list[Operator] = [], probabilities: list[float] = []) -> None:
        """
        Apply a set of operators with associated probabilities to the qubit.
        It usually turns a pure state into a mixed state.

        Args:
            operators: a list of operators, each must operate on a single qubit.
            probabilities: the probability of applying each operator; their sum must be 1.
        """
        i, n = self.state.qubits.index(self), self.state.num
        full_operators: list[Operator] = [op.lift(i, n) for op in operators]
        self.state.stochastic_operate(full_operators, probabilities)

    @override
    def apply_error(self, error) -> None:
        error.qubit(self)

    def __repr__(self) -> str:
        if self.name is not None:
            return "<qubit " + self.name + ">"
        return super().__repr__()
