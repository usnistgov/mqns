#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2024-2025 Amar Abane
#    NIST
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

from qns.entity.node.qnode import QNode
from enum import Enum, auto



class UnexpectedTransitionException(Exception):
    """
    The exception that the FSM transition is incorrect
    """
    pass

class QubitState(Enum):
    ENTANGLED = auto()
    PURIF = auto()
    PENDING = auto()
    ELIGIBLE = auto()
    RELEASE = auto()

class QubitFSM:
    def __init__(self):
        self.state = QubitState.RELEASE
        self.purif_rounds = 0
        
    def to_entangled(self):
        if self.state == QubitState.RELEASE:
            self.state = QubitState.ENTANGLED
            self.purif_rounds = 0
        else:
            print(f"Unexpected transition: <{self.state}> -> <ENTANGLED>")

    def to_purif(self):
        if self.state == QubitState.ENTANGLED:    # swapping conditions met -> go to first purif (if any)
            self.state = QubitState.PURIF
            self.purif_rounds = 0
        elif self.state == QubitState.PENDING:    # pending purif succ -> go to next purif (if any)
            self.state = QubitState.PURIF
            self.purif_rounds +=1
        else:
            print(f"Unexpected transition: <{self.state}> -> <PURIF>")
    
    def to_pending(self):
        if self.state == QubitState.PURIF:
            self.state = QubitState.PENDING
        else:
            print(f"Unexpected transition: <{self.state}> -> <PENDING>")

    def to_release(self):
        if self.state in [QubitState.ENTANGLED, QubitState.PURIF, QubitState.PENDING, QubitState.ELIGIBLE]:
            self.state = QubitState.RELEASE
        else:
            print(f"Unexpected transition: <{self.state}> -> <RELEASE>")
            
    def to_eligible(self):
        if self.state == QubitState.PURIF:
            self.state = QubitState.ELIGIBLE
        else:
            print(f"Unexpected transition: <{self.state}> -> <ELIGIBLE>")

class MemoryQubit():
    """
    An addressable qubit in memory, with a lifecycle.
    """
    def __init__(self, addr: int):
        """
        Args:
            addr (int): address of this qubit in memory
        """
        self.addr = addr
        self.fsm = QubitFSM()
        self.pid = None
        
    def allocate(self, pid: int) -> None:
        self.pid = pid
        
    def deallocate(self) -> None:
        self.pid = None

    def __repr__(self) -> str:
        if self.addr is not None:
            return f"<memory qubit {self.addr}, pid={self.pid}>"
        return super().__repr__()
