#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
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


from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqns.entity.qchannel import QuantumChannel


class QubitState(Enum):
    RAW = auto()
    """
    Qubit is unused.
    """
    ACTIVE = auto()
    """
    The link layer has started a reservation on the qubit as the primary node.
    `qubit.active` contains the reservation key.
    """
    RESERVED = auto()
    """
    Qubit is part of a reservation in link layer and a remote qubit has been found.
    `qubit.active` contains the reservation key.

    This state is set on the qubit at both primary and secondary node of the reservation.
    """
    ENTANGLED0 = auto()
    """
    Qubit is half of an elementary entanglement delivered from link layer.
    `QubitEntangledEvent` has not been processed by forwarder.
    """
    ENTANGLED1 = auto()
    """
    Qubit is half of an elementary entanglement delivered from link layer.
    `QubitEntangledEvent` has been processed by forwarder.
    """
    PURIF = auto()
    """
    Qubit is used by forwarder for zero or more rounds of purification.
    `qubit.qubit_rounds` indicates how many purification rounds have been completed.

    This state is set on the qubit at both primary and secondary node of a purification segment,
    but only the primary node is permitted to initiate purification.
    """
    PENDING = auto()
    """
    The forwarder has initiated purification of the qubit with its partner on a segment.
    `qubit.qubit_rounds` indicates how many purification rounds have been completed, excluding the current round.
    """
    ELIGIBLE = auto()
    """
    Qubit has completed the required rounds of purification and ready for swapping or end-to-end consumption.

    This state is set on the qubit only if own node has a swapping rank no less than the other node in the entanglement.
    """
    RELEASE = auto()
    """
    Qubit is not used by the forwarder.
    The link layer may generate a new elementary entanglement into this qubit.
    """


ALLOWED_STATE_TRANSITIONS: dict[QubitState, tuple[QubitState, ...]] = {
    QubitState.RAW: (QubitState.ACTIVE,),
    QubitState.ACTIVE: (QubitState.RESERVED,),
    QubitState.RESERVED: (QubitState.ENTANGLED0,),
    QubitState.ENTANGLED0: (QubitState.RELEASE, QubitState.ENTANGLED1),
    QubitState.ENTANGLED1: (QubitState.RELEASE, QubitState.PURIF),
    QubitState.PURIF: (QubitState.RELEASE, QubitState.PENDING, QubitState.ELIGIBLE),
    QubitState.PENDING: (QubitState.RELEASE, QubitState.PURIF),
    QubitState.ELIGIBLE: (QubitState.RELEASE,),
    QubitState.RELEASE: (QubitState.RAW,),
}


class PathDirection(Enum):
    LEFT = auto()
    RIGHT = auto()


class MemoryQubit:
    """An addressable qubit in memory, with a lifecycle."""

    def __init__(self, addr: int):
        """Args:
        addr (int): address of this qubit in memory

        """
        self.addr = addr
        """Address index in QuantumMemory."""

        self.qchannel: "QuantumChannel|None" = None
        """qchannel to which qubit is assigned to (currently only at topology creation time)"""
        self.path_id: int | None = None
        """Optional path ID to which qubit is allocated"""
        self.path_direction: PathDirection | None = None
        """Optional end of the path to which the allocated qubit points to (weak solution to avoid loops)"""

        self._state = QubitState.RAW
        """state of the qubit according to the FSM"""
        self.active: str | None = None
        """Reservation key if qubit is reserved for entanglement, None otherwise"""
        self.purif_rounds = 0
        """Number of purification rounds currently completed by the EPR stored on this qubit"""

    def assign(self, ch: "QuantumChannel") -> None:
        self.qchannel = ch

    def unassign(self) -> None:
        self.qchannel = None

    def allocate(self, path_id: int, path_direction: PathDirection | None = None) -> None:
        self.path_id = path_id
        self.path_direction = path_direction

    def deallocate(self) -> None:
        self.path_id = None
        self.path_direction = None

    @property
    def state(self) -> QubitState:
        return self._state

    @state.setter
    def state(self, value: QubitState) -> None:
        if value == self._state:
            return
        if value not in ALLOWED_STATE_TRANSITIONS[self._state]:
            raise ValueError(f"MemoryQubit: unexpected state transition from <{self._state}> to <{value}>; {self}")
        self._state = value

    def reset_state(self) -> None:
        """Reset state to RAW and clear associated fields."""
        self._state = QubitState.RAW
        self.active = None
        self.purif_rounds = 0

    def __repr__(self) -> str:
        return (
            f"<memory qubit {self.addr}, ch={self.qchannel}, path_id={self.path_id}, "
            f"active={self.active}, purif_rounds={self.purif_rounds}, state={self._state}>"
        )
