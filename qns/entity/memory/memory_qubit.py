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

from qns.utils import log

if TYPE_CHECKING:
    from qns.entity.qchannel import QuantumChannel


class QubitState(Enum):
    ENTANGLED = auto()
    PURIF = auto()
    PENDING = auto()
    ELIGIBLE = auto()
    RELEASE = auto()


class QubitFSM:
    def __init__(self):
        self.state = QubitState.RELEASE

    def to_entangled(self):
        if self.state == QubitState.RELEASE:
            self.state = QubitState.ENTANGLED
        else:
            log.debug(f"Unexpected transition: <{self.state}> -> <ENTANGLED>")

    def to_purif(self):
        if self.state in (QubitState.ENTANGLED, QubitState.PENDING):
            # from ENTANGLED: swapping conditions met -> go to first purif (if any)
            # from PENDING: pending purif succ -> go to next purif (if any)
            self.state = QubitState.PURIF
        else:
            log.debug(f"Unexpected transition: <{self.state}> -> <PURIF>")

    def to_pending(self):
        if self.state == QubitState.PURIF:
            self.state = QubitState.PENDING
        else:
            log.debug(f"Unexpected transition: <{self.state}> -> <PENDING>")

    def to_release(self):
        if self.state in (QubitState.ENTANGLED, QubitState.PURIF, QubitState.PENDING, QubitState.ELIGIBLE):
            self.state = QubitState.RELEASE
        else:
            log.debug(f"Unexpected transition: <{self.state}> -> <RELEASE>")

    def to_eligible(self):
        if self.state == QubitState.PURIF:
            self.state = QubitState.ELIGIBLE
        else:
            log.debug(f"Unexpected transition: <{self.state}> -> <ELIGIBLE>")

    def __repr__(self) -> str:
        return f"{self.state}"


class MemoryQubit:
    """An addressable qubit in memory, with a lifecycle."""

    def __init__(self, addr: int):
        """Args:
        addr (int): address of this qubit in memory

        """
        self.addr = addr
        self.fsm = QubitFSM()
        """state of the qubit according to the FSM"""
        self.qchannel: "QuantumChannel|None" = None
        """qchannel to which qubit is assigned to (currently only at topology creation time)"""
        self.path_id: int | None = None
        """Optional path ID to which qubit is allocated"""
        self.active: str | None = None
        """Reservation key if qubit is reserved for entanglement, None otherwise"""
        self.purif_rounds = 0
        """Number of purification rounds currently completed by the EPR stored on this qubit"""

    def allocate(self, path_id: int) -> None:
        self.path_id = path_id

    def deallocate(self) -> None:
        self.path_id = None

    def assign(self, ch: "QuantumChannel") -> None:
        self.qchannel = ch

    def unassign(self) -> None:
        self.qchannel = None

    def __repr__(self) -> str:
        if self.addr is not None:
            return (
                f"<memory qubit {self.addr}, ch={self.qchannel}, path_id={self.path_id}, "
                + f"active={self.active}, purif_rounds={self.purif_rounds}, state={self.fsm}>"
            )
        return super().__repr__()
