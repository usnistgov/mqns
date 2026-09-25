import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, overload, override

from mqns.simulator import Event, Time, event_handler, func_to_event
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.network import QuantumNetwork


class TimingPhase(Enum):
    EXTERNAL = auto()
    ROUTING = auto()
    INTERNAL = auto()


class TimingMode(ABC):
    """
    Network-wide application timing mode.
    """

    def __init__(self, name: str):
        self.name = name

    def install(self, network: "QuantumNetwork"):
        self.simulator = network.simulator
        self.network = network

    @abstractmethod
    def is_async(self) -> bool:
        """
        Determine whether the network is using ASYNC timing.
        """
        pass

    @abstractmethod
    def _is_phase(self, phase: TimingPhase, t: Time | None = None) -> bool: ...

    def is_external(self, t: Time | None = None) -> bool:
        """
        Determine whether the network is either using ASYNC timing or in an EXTERNAL phase.

        Args:
            t: If specified, also check that the timestamp is in the same phase window.
        """
        return self._is_phase(TimingPhase.EXTERNAL, t)

    def is_routing(self, t: Time | None = None) -> bool:
        """
        Determine whether the network is either using ASYNC timing or in a ROUTING phase.

        Args:
            t: If specified, also check that the timestamp is in the same phase window.
        """
        return self._is_phase(TimingPhase.ROUTING, t)

    def is_internal(self, t: Time | None = None) -> bool:
        """
        Determine whether the network is either using ASYNC timing or in an INTERNAL phase.

        Args:
            t: If specified, also check that the timestamp is in the same phase window.
        """
        return self._is_phase(TimingPhase.INTERNAL, t)


class TimingModeAsync(TimingMode):
    """
    Asynchronous application timing mode.
    """

    def __init__(self, *, name="ASYNC"):
        super().__init__(name)

    @override
    def install(self, network: "QuantumNetwork"):
        super().install(network)
        log.info("TIME_SYNC: using %s mode", self.name)

    @override
    def is_async(self) -> bool:
        return True

    @override
    def _is_phase(self, phase: TimingPhase, t: Time | None = None) -> bool:
        _ = phase, t
        return True


class TimingModeSync(TimingMode):
    """
    Synchronous application timing mode.
    """

    @overload
    def __init__(
        self,
        *,
        name="SYNC",
        t_ext: float,
        t_rtg: float = 0,
        t_int: float,
    ):
        """
        Args:
            t_ext: EXTERNAL phase duration in seconds.
            t_rtg: ROUTING phase duration in seconds, defaults to zero.
            t_int: INTERNAL phase duration in seconds.
        """

    @overload
    def __init__(
        self,
        *,
        name="SYNC",
        durations: Sequence[float],
    ):
        """
        Args:
            durations: EXTERNAL, ROUTING, INTERNAL phase durations in seconds.
        """

    def __init__(
        self,
        *,
        name="SYNC",
        t_ext: float = 0,
        t_rtg: float = 0,
        t_int: float = 0,
        durations: Sequence[float] | None = None,
    ):
        super().__init__(name)

        self._seconds = (t_ext, t_rtg, t_int) if durations is None else durations
        if len(self._seconds) != 3:
            raise ValueError("durations= must have exactly three values")

    @override
    def install(self, network: "QuantumNetwork"):
        super().install(network)

        t_ext, t_rtg, t_int = self._seconds
        if t_ext <= 0:
            raise ValueError("EXTERNAL phase duration must be positive")
        if t_rtg < 0:
            raise ValueError("ROUTING phase duration must be non-negative")
        if t_int <= 0:
            raise ValueError("INTERNAL phase duration must be positive")

        self.t_ext = self.simulator.time(sec=t_ext)
        """EXTERNAL phase duration."""
        self.t_rtg = self.simulator.time(sec=t_rtg)
        """ROUTING phase duration."""
        self.t_int = self.simulator.time(sec=t_int)
        """INTERNAL phase duration."""

        log.info(
            "TIME_SYNC: using %s mode, t_ext=%s, t_rtg=%s, t_int=%s",
            self.name,
            self.t_ext.slot,
            self.t_rtg.slot,
            self.t_int.slot,
        )

        self._sequence = [
            (phase, duration, _PHASE_EVENTS[phase, True], _PHASE_EVENTS[phase, False])
            for (phase, duration) in [
                (TimingPhase.EXTERNAL, self.t_ext),
                (TimingPhase.ROUTING, self.t_rtg),
                (TimingPhase.INTERNAL, self.t_int),
            ]
            if duration.slot > 0
        ]
        self._pos = -1

        self.end_time = self.simulator.ts
        """Current phase end time (exclusive)."""

        self.simulator.sched(func_to_event(self.simulator.ts, self._enter_phase))

    @property
    def phase(self) -> TimingPhase:
        """Current phase."""
        return self._sequence[self._pos][0]

    def _broadcast_event(self, EventType: type[Event]) -> None:
        event = EventType(self.simulator.tc)
        for node in self.network.all_nodes:
            node.handle(event)

    def _change_phase(self):
        phase, _, _, ExitEvent = self._sequence[self._pos]
        log.debug("TIME_SYNC: exiting %s phase", phase.name)

        self._broadcast_event(ExitEvent)

        self._enter_phase()

    def _enter_phase(self):
        self._pos += 1
        if self._pos >= len(self._sequence):
            self._pos = 0

        phase, duration, EnterEvent, _ = self._sequence[self._pos]
        log.debug("TIME_SYNC: entering %s phase", phase.name)

        self.end_time = self.simulator.tc + duration
        self.simulator.sched(func_to_event(self.end_time, self._change_phase))

        self._broadcast_event(EnterEvent)

    @override
    def is_async(self) -> bool:
        return False

    @override
    def _is_phase(self, phase: TimingPhase, t: Time | None = None) -> bool:
        return self.phase is phase and (t is None or t < self.end_time)


def _phase_event_invoke(self: Event) -> None:
    _ = self
    # Phase events are directly dispatched onto nodes without going through the scheduler
    # for performance reasons, so that the invoke() method is unused.
    raise RuntimeError("TimingPhaseEvent.invoke() is unused")


def _phase_event_cancel(self: Event) -> None:
    _ = self
    # Phase events must be dispatched to every application on a node,
    # so that cancellation is disallowed.
    raise RuntimeError("TimingPhaseEvent cannot be canceled")


_PHASE_EVENTS: dict[tuple[TimingPhase, bool], type[Event]] = {}
for phase, enter in itertools.product(TimingPhase, (False, True)):
    _PHASE_EVENTS[phase, enter] = type(
        f"TimingPhaseEvent{'Enter' if enter else 'Exit'}{phase.name}",
        (Event,),
        {"invoke": _phase_event_invoke, "cancel": _phase_event_cancel},
    )


def sync_phase_handler(phase: TimingPhase, enter: bool):
    """
    Method decorator for entering or exiting a timing phase in SYNC timing mode.

    Args:
        phase: Timing phase.
        enter: True for entering, False for exiting.
    """

    def decorator(f: Callable[[Any], None]):
        def sync_phase_handler_wrapper(self: Any, _: Event) -> None:
            f(self)

        h = event_handler(_PHASE_EVENTS[phase, enter])(sync_phase_handler_wrapper)
        h.__doc__ = f.__doc__ or f"In SYNC timing mode, {'enter' if enter else 'exit'} {phase.name} phase."
        return h

    return decorator
