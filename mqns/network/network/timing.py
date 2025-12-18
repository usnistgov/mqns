from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, final, override

from mqns.simulator import Event, Time, func_to_event
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.network import QuantumNetwork


class TimingPhase(Enum):
    EXTERNAL = auto()
    INTERNAL = auto()


@final
class TimingPhaseEvent(Event):
    """
    Event that indicates a timing phase change, emitted in SYNC timing mode only.
    """

    def __init__(self, phase: TimingPhase, *, t: Time, name: str | None = None, by: Any = None):
        super().__init__(t, name, by)
        self.phase = phase

    @override
    def invoke(self) -> None:
        # This event is directly dispatched onto nodes without going through the scheduler
        # for performance reasons, so that the invoke() method is unused.
        raise RuntimeError


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
    def is_external(self, t: Time | None = None) -> bool:
        """
        Determine whether the network is either using ASYNC timing or in an EXTERNAL phase.

        Args:
            t: If specified, also check that the timestamp is in the same phase window.
        """
        pass

    @abstractmethod
    def is_internal(self, t: Time | None = None) -> bool:
        """
        Determine whether the network is either using ASYNC timing or in an INTERNAL phase.

        Args:
            t: If specified, also check that the timestamp is in the same phase window.
        """
        pass


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
    def is_external(self, t: Time | None = None) -> bool:
        _ = t
        return True

    @override
    def is_internal(self, t: Time | None = None) -> bool:
        _ = t
        return True


class TimingModeSync(TimingMode):
    """
    Synchronous application timing mode.
    """

    def __init__(self, *, name="SYNC", t_ext: float, t_int: float):
        """
        Args:
            t_ext: EXTERNAL phase duration.
            t_int: INTERNAL phase duration.
        """
        super().__init__(name)
        self.sequence = deque[tuple[TimingPhase, float]](
            [
                (TimingPhase.EXTERNAL, t_ext),
                (TimingPhase.INTERNAL, t_int),
            ]
        )
        self.phase = self.sequence[-1][0]
        """Current phase."""
        self.end_time = Time()
        """Current phase end time (exclusive)."""

    @override
    def install(self, network: "QuantumNetwork"):
        super().install(network)
        self.simulator.add_event(func_to_event(self.simulator.ts, self.signal_phase, by=self))

    def signal_phase(self):
        this_phase = self.sequence.popleft()
        self.sequence.append(this_phase)
        phase, duration = this_phase

        self.phase = phase
        self.end_time = self.simulator.tc + duration

        # schedule next sync signal
        self.simulator.add_event(func_to_event(self.end_time, self.signal_phase, by=self))

        log.debug(f"TIME_SYNC: signal {phase.name} phase")
        event = TimingPhaseEvent(phase, t=self.simulator.tc)
        for node in self.network.all_nodes:
            node.handle(event)

    @override
    def is_async(self) -> bool:
        return False

    @override
    def is_external(self, t: Time | None = None) -> bool:
        return self.phase == TimingPhase.EXTERNAL and (t is None or t < self.end_time)

    @override
    def is_internal(self, t: Time | None = None) -> bool:
        return self.phase == TimingPhase.INTERNAL and (t is None or t < self.end_time)
