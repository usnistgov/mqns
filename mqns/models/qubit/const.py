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

import numpy as np

from mqns.models.qubit.typing import Basis, Operator1, Operator2, QubitState

QUBIT_STATE_0: QubitState = np.array([[1], [0]], dtype=np.complex128)
QUBIT_STATE_1: QubitState = np.array([[0], [1]], dtype=np.complex128)

QUBIT_STATE_P: QubitState = 1 / np.sqrt(2) * np.array([[1], [1]], dtype=np.complex128)
QUBIT_STATE_N: QubitState = 1 / np.sqrt(2) * np.array([[1], [-1]], dtype=np.complex128)

QUBIT_STATE_R: QubitState = 1 / np.sqrt(2) * np.array([[-1j], [1]], dtype=np.complex128)
QUBIT_STATE_L: QubitState = 1 / np.sqrt(2) * np.array([[1], [-1j]], dtype=np.complex128)

OPERATOR_HADAMARD: Operator1 = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
OPERATOR_T: Operator1 = np.array([[1, 0], [0, np.e ** (1j * np.pi / 4)]], dtype=np.complex128)
OPERATOR_S: Operator1 = np.array([[1, 0], [0, 1j]], dtype=np.complex128)

OPERATOR_PAULI_I: Operator1 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
OPERATOR_PAULI_X: Operator1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
OPERATOR_PAULI_Y: Operator1 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
OPERATOR_PAULI_Z: Operator1 = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def OPERATOR_RX(theta: float) -> Operator1:
    return np.array(
        [[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.complex128
    )


def OPERATOR_RY(theta: float) -> Operator1:
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.complex128)


def OPERATOR_RZ(theta: float) -> Operator1:
    return np.array([[np.e ** (-0.5j * theta), 0], [0, np.e ** (0.5j * theta)]], dtype=np.complex128)


def OPERATOR_PHASE_SHIFT(theta: float) -> Operator1:
    return np.array([[1, 0], [0, np.e ** (1j * theta)]], dtype=np.complex128)


OPERATOR_CNOT: Operator2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
OPERATOR_SWAP: Operator2 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)

BASIS_Z: Basis = OPERATOR_PAULI_Z
BASIS_X: Basis = OPERATOR_PAULI_X
BASIS_Y: Basis = OPERATOR_PAULI_Y
