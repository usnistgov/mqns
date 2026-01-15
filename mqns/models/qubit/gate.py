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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from mqns.models.qubit.operator import (
    OPERATOR_H,
    OPERATOR_PAULI_I,
    OPERATOR_PAULI_X,
    OPERATOR_PAULI_Y,
    OPERATOR_PAULI_Z,
    OPERATOR_PHASE_SHIFT,
    OPERATOR_RX,
    OPERATOR_RY,
    OPERATOR_RZ,
    OPERATOR_S,
    OPERATOR_T,
    Operator,
)

if TYPE_CHECKING:
    from mqns.models.qubit.qubit import Qubit


class Gate:
    """The quantum gates that will operate qubits"""

    def __init__(self, name: str, _docs: str | None = None):
        """Args:
        name: the gate's name

        """
        self._name = name
        self.__doc__ = _docs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class SingleQubitGate(Gate):
    """The single qubit gates operate on a single qubit"""

    pass


class SingleQubitSimpleGate(SingleQubitGate):
    def __init__(self, name: str, operator: Operator, _docs: str | None = None):
        """
        Args:
            name: the gate's name
            operator: the matrix represent of this operator
        """
        super().__init__(name, _docs)
        self._operator = operator

    def __call__(self, qubit: "Qubit") -> None:
        """
        Args:
            qubit (Qubit): the operating qubit
        """
        qubit.operate(self._operator)


X = SingleQubitSimpleGate(name="X", operator=OPERATOR_PAULI_X, _docs="Pauli X Gate")
Y = SingleQubitSimpleGate(name="Y", operator=OPERATOR_PAULI_Y, _docs="Pauli Y Gate")
Z = SingleQubitSimpleGate(name="Z", operator=OPERATOR_PAULI_Z, _docs="Pauli Z Gate")
I = SingleQubitSimpleGate(name="I", operator=OPERATOR_PAULI_I, _docs="Pauli I Gate")
H = SingleQubitSimpleGate(name="H", operator=OPERATOR_H, _docs="Hadamard Gate")
T = SingleQubitSimpleGate(name="T", operator=OPERATOR_T, _docs="T gate (pi/4 shift gate)")
S = SingleQubitSimpleGate(name="S", operator=OPERATOR_S, _docs="S gate (pi/2 shift gate)")


class SingleQubitRotateGate(SingleQubitGate):
    def __init__(self, name: str, operator: Callable[[float], Operator], _docs: str | None = None):
        """
        Args:
            name: the gate's name
            operator: a function that returns the matrix represent of this operator
        """
        super().__init__(name, _docs)
        self._operator = operator

    def __call__(self, qubit: "Qubit", theta=np.pi / 4) -> None:
        """
        Args:
            qubit (Qubit): the operating qubit
            theta (float): the rotating degree
        """
        qubit.operate(self._operator(theta))


R = SingleQubitRotateGate(name="R", operator=OPERATOR_PHASE_SHIFT, _docs="R gate (phase shift gate)")
RX = SingleQubitRotateGate(name="RX", operator=OPERATOR_RX, _docs="Rx gate (X rotate gate)")
RY = SingleQubitRotateGate(name="RY", operator=OPERATOR_RY, _docs="Ry gate (Y rotate gate)")
RZ = SingleQubitRotateGate(name="RZ", operator=OPERATOR_RZ, _docs="Rz gate (Z rotate gate)")


class SingleQubitArbitraryGate(SingleQubitGate):
    def __call__(self, qubit: "Qubit", operator: Operator) -> None:
        """
        Args:
            qubit (Qubit): the operating qubit
            operator: the operator matrix
        """
        if operator.n != 1:
            raise ValueError("wrong operator size")
        self._operator = operator
        qubit.operate(self._operator)


U = SingleQubitArbitraryGate(name="U", _docs="Arbitrary single qubit operation gate")


class DoubleQubitsControlledGate(Gate):
    """The double qubits gates operate on two qubits, including a controlled qubit and a operating qubit.

    The controlled  gate:

        [[I_2, 0][0, operator]]
    """

    def __init__(self, name: str, operator: Operator, _docs: str | None = None):
        """
        Args:
            name: the gate's name
            operator: the matrix represent of the operator
        """
        super().__init__(name, _docs)
        self._operator = operator

    def __call__(self, qubit1: "Qubit", qubit2: "Qubit", operator: Operator | None = None) -> None:
        """
        Args:
            qubit1: the first qubit (controller)
            qubit2: the second qubit
            operator: the matrix represent of the operator
        """
        from mqns.models.qubit.qubit import QState  # noqa: PLC0415

        if operator is None:
            operator = self._operator
        if operator.n != 1:
            raise ValueError("wrong operator size")

        if qubit1 == qubit2:
            return
        state = QState.joint(qubit1, qubit2)

        idx1 = state.qubits.index(qubit1)
        idx2 = state.qubits.index(qubit2)

        full_operator_part_0 = np.array([1])  # |0> <0|
        full_operator_part_1 = np.array([1])  # |1> <1|

        for i in range(state.num):
            if i == idx1:
                full_operator_part_0 = np.kron(full_operator_part_0, np.array([[1, 0], [0, 0]]))
                full_operator_part_1 = np.kron(full_operator_part_1, np.array([[0, 0], [0, 1]]))
            elif i == idx2:
                full_operator_part_0 = np.kron(full_operator_part_0, OPERATOR_PAULI_I.u)
                full_operator_part_1 = np.kron(full_operator_part_1, operator.u)
            else:
                full_operator_part_0 = np.kron(full_operator_part_0, OPERATOR_PAULI_I.u)
                full_operator_part_1 = np.kron(full_operator_part_1, OPERATOR_PAULI_I.u)
        full_operator = full_operator_part_0 + full_operator_part_1
        qubit1.state.operate(Operator(full_operator, state.num))


ControlledGate = DoubleQubitsControlledGate(name="Controlled Gate", operator=OPERATOR_PAULI_X, _docs="The controlled gate")
CNOT = DoubleQubitsControlledGate(name="Controlled NOT Gate", operator=OPERATOR_PAULI_X, _docs="The controlled Pauli-X gate")
CX = DoubleQubitsControlledGate(name="Controlled Pauli-X Gate", operator=OPERATOR_PAULI_X, _docs="The controlled Pauli-X gate")
CY = DoubleQubitsControlledGate(name="Controlled Pauli-Y Gate", operator=OPERATOR_PAULI_Y, _docs="The controlled Pauli-Y gate")
CZ = DoubleQubitsControlledGate(name="Controlled Pauli-Z Gate", operator=OPERATOR_PAULI_Z, _docs="The controlled Pauli-Z gate")


class DoubleQubitsRotateGate(Gate):
    def __init__(self, name: str, operator: Callable[[float], Operator], _docs: str | None = None):
        """
        Args:
            name: the gate's name
            operator: a function that returns the matrix represent of the operator
        """
        super().__init__(name, _docs)
        self._operator = operator

    def __call__(self, qubit1: "Qubit", qubit2: "Qubit", theta: float = np.pi / 4) -> None:
        operator = self._operator(theta)
        super().__call__(qubit1, qubit2, operator=operator)


CR = DoubleQubitsRotateGate(
    name="Controlled Phase Rotate Gate", operator=OPERATOR_PHASE_SHIFT, _docs="The controlled rotate gate"
)


class SwapGate(Gate):
    def __call__(self, qubit1: "Qubit", qubit2: "Qubit"):
        """
        The swap gate, swap the states of qubit1 and qubit2

        Args:
            qubit1 (Qubit): the first qubit (controller)
            qubit2 (Qubit): the second qubit
        """
        from mqns.models.qubit.qubit import QState  # noqa: PLC0415

        if qubit1 == qubit2:
            return
        state = QState.joint(qubit1, qubit2)
        idx1 = state.qubits.index(qubit1)
        idx2 = state.qubits.index(qubit2)

        state.qubits[idx1], state.qubits[idx2] = state.qubits[idx2], state.qubits[idx1]


Swap = SwapGate(name="Swap Gate", _docs="swap the states of qubit1 and qubit2")


class ThreeQubitsGate(Gate):
    """
    The gate operates on three qubits, including 2 controlled qubit and a operating qubit.

    The 3 controlled-controlled gate:

        [[I_6, 0][0, operator]]
    """

    def __init__(self, name: str, operator: Operator = OPERATOR_PAULI_X, _docs: str | None = None):
        """
        Args:
            name: the gate's name
            operator: the matrix represent of the operator
        """
        super().__init__(name, _docs)
        self._operator = operator

    def __call__(self, qubit1: "Qubit", qubit2: "Qubit", qubit3: "Qubit", operator: Operator | None = None) -> Any:
        from mqns.models.qubit.qubit import QState  # noqa: PLC0415

        if operator is None:
            operator = self._operator
        if operator.n != 1:
            raise ValueError("wrong operator size")

        if qubit1 == qubit2 or qubit1 == qubit3 or qubit2 == qubit3:  # noqa: PLR1714
            return
        QState.joint(qubit1, qubit2)
        state = QState.joint(qubit2, qubit3)

        # single qubit operate
        idx1 = state.qubits.index(qubit1)
        idx2 = state.qubits.index(qubit2)
        idx3 = state.qubits.index(qubit3)

        full_operator_part_00 = np.array([1])  # |0> <0|
        full_operator_part_01 = np.array([1])  # |1> <1|
        full_operator_part_10 = np.array([1])  # |0> <0|
        full_operator_part_11 = np.array([1])  # |1> <1|

        for i in range(state.num):
            if i == idx1:
                full_operator_part_00 = np.kron(full_operator_part_00, np.array([[1, 0], [0, 0]]))
                full_operator_part_01 = np.kron(full_operator_part_01, np.array([[1, 0], [0, 0]]))
                full_operator_part_10 = np.kron(full_operator_part_10, np.array([[0, 0], [0, 1]]))
                full_operator_part_11 = np.kron(full_operator_part_11, np.array([[0, 0], [0, 1]]))
            elif i == idx2:
                full_operator_part_00 = np.kron(full_operator_part_00, np.array([[1, 0], [0, 0]]))
                full_operator_part_10 = np.kron(full_operator_part_10, np.array([[1, 0], [0, 0]]))
                full_operator_part_01 = np.kron(full_operator_part_01, np.array([[0, 0], [0, 1]]))
                full_operator_part_11 = np.kron(full_operator_part_11, np.array([[0, 0], [0, 1]]))
            elif i == idx3:
                full_operator_part_00 = np.kron(full_operator_part_00, OPERATOR_PAULI_I.u)
                full_operator_part_01 = np.kron(full_operator_part_01, OPERATOR_PAULI_I.u)
                full_operator_part_10 = np.kron(full_operator_part_10, OPERATOR_PAULI_I.u)
                full_operator_part_11 = np.kron(full_operator_part_11, operator.u)
            else:
                full_operator_part_00 = np.kron(full_operator_part_00, OPERATOR_PAULI_I.u)
                full_operator_part_01 = np.kron(full_operator_part_01, OPERATOR_PAULI_I.u)
                full_operator_part_10 = np.kron(full_operator_part_10, OPERATOR_PAULI_I.u)
                full_operator_part_11 = np.kron(full_operator_part_11, OPERATOR_PAULI_I.u)
        full_operator = full_operator_part_00 + full_operator_part_01 + full_operator_part_10 + full_operator_part_11
        qubit1.state.operate(Operator(full_operator, state.num))


Toffoli = ThreeQubitsGate(name="Toffoli Gate", operator=OPERATOR_PAULI_X, _docs="The controlled-controlled (Toffoli) gate")
