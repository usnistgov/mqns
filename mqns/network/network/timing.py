from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from enum import Enum, auto
from typing import TYPE_CHECKING, final, overload, override

from mqns.simulator import Event, Time, func_to_event
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.network import QuantumNetwork


class TimingPhase(Enum):
    EXTERNAL = auto()
    ROUTING = auto()
    INTERNAL = auto()


@final
class TimingPhaseEvent(Event):
    """
    Event that indicates a timing phase change, emitted in SYNC timing mode only.
    """

    def __init__(self, phase: TimingPhase, *, enter: bool, t: Time, name: str | None = None):
        super().__init__(t, name)
        self.phase = phase
        """Phase."""
        self.enter = enter
        """True when entering a phase, False when exiting a phase."""

    @override
    def invoke(self) -> None:
        # This event is directly dispatched onto nodes without going through the scheduler
        # for performance reasons, so that the invoke() method is unused.
        raise RuntimeError

    @property
    def action(self) -> tuple[TimingPhase, bool]:
        return self.phase, self.enter


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

        self._sequence = deque[tuple[TimingPhase, Time]]()
        self._sequence.append((TimingPhase.EXTERNAL, self.t_ext))
        if t_rtg > 0:
            self._sequence.append((TimingPhase.ROUTING, self.t_rtg))
        self._sequence.append((TimingPhase.INTERNAL, self.t_int))

        self.phase = self._sequence[-1][0]
        """Current phase."""
        self.end_time = self.simulator.ts
        """Current phase end time (exclusive)."""

        self.simulator.add_event(func_to_event(self.simulator.ts, self._enter_phase))

    def _change_phase(self):
        log.debug(f"TIME_SYNC: exiting {self.phase.name} phase")
        event = TimingPhaseEvent(self.phase, enter=False, t=self.simulator.tc)
        for node in self.network.all_nodes:
            node.handle(event)

        self._enter_phase()

    def _enter_phase(self):
        this_phase = self._sequence.popleft()
        self._sequence.append(this_phase)
        phase, duration = this_phase

        self.phase = phase
        self.end_time = self.simulator.tc + duration

        # schedule next sync signal
        self.simulator.add_event(func_to_event(self.end_time, self._change_phase))

        log.debug(f"TIME_SYNC: entering {phase.name} phase")
        event = TimingPhaseEvent(phase, enter=True, t=self.simulator.tc)
        for node in self.network.all_nodes:
            node.handle(event)

    @override
    def is_async(self) -> bool:
        return False

    @override
    def _is_phase(self, phase: TimingPhase, t: Time | None = None) -> bool:
        return self.phase is phase and (t is None or t < self.end_time)
