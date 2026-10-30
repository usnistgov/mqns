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


from collections.abc import Callable, Iterable
from enum import Enum, auto
from typing import Any

from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.simulator import EventHandleSet, Time


class QubitState(Enum):
    """
    Indicates qubit state following a finite state machine.

    .. figure:: /_static/qubit-state.svg
       :alt: finite state machine
       :align: center
       :width: 100%
    """

    RAW = auto()
    """
    Qubit is unused.
    """
    ACTIVE = auto()
    """
    The link layer has requested a reservation on the qubit as the primary node.
    ``qubit.active`` contains the reservation key.
    """
    RESERVED = auto()
    """
    Qubit is part of a reservation in link layer and a remote qubit has been found.
    ``qubit.active`` contains the reservation key.

    This state is set on the qubit at both primary and secondary node of the reservation.
    """
    ENTANGLED0 = auto()
    """
    Qubit is half of an elementary entanglement delivered from link layer.
    ``QubitEntangledEvent`` has not been processed by forwarder.
    """
    ENTANGLED1 = auto()
    """
    Qubit is half of an elementary entanglement delivered from link layer.
    ``QubitEntangledEvent`` has been processed by forwarder.
    """
    PURIF = auto()
    """
    Qubit is used by forwarder for zero or more rounds of purification.
    ``qubit.qubit_rounds`` indicates how many purification rounds have been completed.

    This state is set on the qubit at both primary and secondary node of a purification segment,
    but only the primary node is permitted to initiate purification.
    """
    PENDING = auto()
    """
    The forwarder has initiated purification of the qubit with its partner on a segment.
    ``qubit.qubit_rounds`` indicates how many purification rounds have been completed, excluding the current round.
    """
    ELIGIBLE = auto()
    """
    Qubit has completed the required rounds of purification and ready for swapping or end-to-end consumption.

    This state is set on the qubit only if own node has a swapping rank no less than the other node in the entanglement.
    """
    SWAPPING = auto()
    """
    The forwarder is performing local swapping between this and another memory qubit.
    If the qubit is released from this state, the local swapping would be aborted.
    """
    CONSUME = auto()
    """
    Qubit is part of an end-to-end entangled pair and ready for consumption by application.
    """
    RELEASE = auto()
    """
    Qubit is not used by the forwarder.
    The link layer may generate a new elementary entanglement into this qubit.
    """


ALLOWED_STATE_TRANSITIONS: dict[QubitState, tuple[QubitState, ...]] = {
    QubitState.RAW: (QubitState.ACTIVE, QubitState.RESERVED),
    QubitState.ACTIVE: (QubitState.RAW, QubitState.RESERVED),
    QubitState.RESERVED: (QubitState.RAW, QubitState.ENTANGLED0),
    QubitState.ENTANGLED0: (QubitState.RELEASE, QubitState.ENTANGLED1),
    QubitState.ENTANGLED1: (QubitState.RELEASE, QubitState.PURIF),
    QubitState.PURIF: (QubitState.RELEASE, QubitState.PENDING, QubitState.ELIGIBLE),
    QubitState.PENDING: (QubitState.RELEASE, QubitState.PURIF),
    QubitState.ELIGIBLE: (QubitState.SWAPPING, QubitState.CONSUME, QubitState.RELEASE),
    QubitState.SWAPPING: (QubitState.RELEASE,),
    QubitState.CONSUME: (QubitState.RELEASE,),
    QubitState.RELEASE: (QubitState.RAW,),
}
"""
Allowed state transitions, mapped from old state to allowed new states.
"""


class PathDirection(Enum):
    L = 0
    """
    Path direction LEFT: qubit is assigned to a channel that connects to the left side neighbor.
    """
    R = 1
    """
    Path direction RIGHT: qubit is assigned to a channel that connects to the right side neighbor.
    """


class MemoryQubit:
    """An addressable qubit in memory, with a lifecycle."""

    addr: int
    """Address index in QuantumMemory."""

    qchannel: QuantumChannel | None = None
    """QuantumChannel to which qubit is assigned to."""
    path_id: int | None = None
    """Optional path ID to which qubit is allocated."""
    path_direction: PathDirection | None = None
    """Optional end of the path to which the allocated qubit points to (weak solution to avoid loops)."""

    _state = QubitState.RAW

    key: str | None = None
    """
    Qubit reservation key, used during entanglement.

    This is set by ``LinkLayer`` and remains unchanged until the qubit is released.
    It is a consistent identifier for a qubit that contains a particular qstate and its swapped/purified forms.
    """
    partner: tuple[QNode, str] | None = None
    """
    Partner node and partner qubit reservation key, if the qubit contains an entanglement.

    This is set by ``Forwarder`` upon entanglement and when completing a swap rank sequentially.
    It is invalid during parallel swap within a rank.
    """
    epr_path_ids: list[int] | None = None
    """
    Which paths are compatible with the currently stored entanglement.

    Note: this is a ``list`` instead of a ``set``, to ensure deterministic simulation.
    """
    purif_rounds = 0
    """Number of purification rounds completed by the EPR stored on this qubit."""
    eligible_time = Time.SENTINEL
    """
    Time point when the qubit becomes eligible.

    This is set by ``Forwarder`` upon processing the ELIGIBLE state transition.
    It is invalid if qubit state is not ELIGIBLE/SWAPPING/CONSUME.
    """

    events: EventHandleSet
    """Events that are canceled upon reaching RELEASE state."""

    _on_raw_callbacks: list[Callable[["MemoryQubit"], Any]] | None = None

    def __init__(self, addr: int):
        self.addr = addr
        self.events = EventHandleSet()

    @property
    def state(self) -> QubitState:
        """
        Retrieve or set qubit state.

        In the setter:

        * If the state transition is not allowed, raises ValueError,
        * If the new state is either ``RAW`` or ``RELEASE``, cancel events.
        * If the new state is ``RAW``, clear LinkLayer and Forwarder related fields.
        """
        return self._state

    @state.setter
    def state(self, value: QubitState) -> None:
        if value is self._state:
            return
        if value not in ALLOWED_STATE_TRANSITIONS[self._state]:
            raise ValueError(f"MemoryQubit: unexpected state transition from <{self._state}> to <{value}>; {self}")

        if value in (QubitState.RAW, QubitState.RELEASE):
            self.reset_state(value)
        else:
            self._state = value

    def reset_state(self, state: QubitState) -> None:
        """Reset state to RELEASE/RAW and clear associated fields."""
        self._state = state
        self.events.clear()
        if state is not QubitState.RAW:
            return

        self.key = None
        self.partner = None
        self.epr_path_ids = None
        self.purif_rounds = 0
        self.eligible_time = Time.SENTINEL

        if not self._on_raw_callbacks:
            return
        for fn in self._on_raw_callbacks:
            fn(self)
        del self._on_raw_callbacks

    def on_raw(self, fn: Callable[["MemoryQubit"], Any]) -> None:
        """
        Schedule a function to be invoked when the qubit reaches RAW state.
        If the qubit is already in RAW state, the function is called immediately.
        """
        if self._state is QubitState.RAW:
            fn(self)
        else:
            if not self._on_raw_callbacks:
                self._on_raw_callbacks = []
            self._on_raw_callbacks.append(fn)

    def __repr__(self) -> str:
        return ", ".join(_describe(self)) + ")"


def _describe(mq: MemoryQubit) -> Iterable[str]:
    yield f"MemoryQubit({mq.addr}"
    yield f"state={mq._state.name}"

    if mq.qchannel:
        yield f"ch={mq.qchannel.name}"
        if mq.path_direction:
            yield f"path={mq.path_id}-{mq.path_direction.name}"

    if mq.key:
        yield f"key={mq.key}"
    if mq.partner:
        yield f"partner={mq.partner[0].name}:{mq.partner[1]}"

    if mq.epr_path_ids:
        yield f"epr-path-ids={mq.epr_path_ids}"

    if mq._state in (QubitState.PURIF, QubitState.PENDING, QubitState.ELIGIBLE, QubitState.SWAPPING, QubitState.CONSUME):
        yield f"purif_rounds={mq.purif_rounds}"
