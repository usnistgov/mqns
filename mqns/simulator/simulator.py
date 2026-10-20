import math
import os
import time
from collections.abc import Callable, Iterable
from pstats import SortKey
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Protocol, overload

from mqns.simulator.event import Event, func_to_event
from mqns.simulator.pool import HeapEventPool, SynchronizedEventPool
from mqns.simulator.time import Time
from mqns.utils import log

try:
    from cProfile import Profile
except ImportError:
    from profile import Profile

if TYPE_CHECKING:
    from mqns.entity.monitor import Monitor


class SimulatorOwner(Protocol):
    simulator: "Simulator"


class SimulatorInstallable(Protocol):
    def install(self, simulator: "Simulator") -> Any: ...


class Simulator:
    """
    Discrete-event driven simulator core.
    """

    _run_tracebackhide: ClassVar = True
    """
    If True, the ``.run()`` method is hidden from pytest traceback.
    """

    watchers: dict[type[Event], list["Monitor"]] | None = None

    @staticmethod
    def ensure_not_installed_to(obj: SimulatorOwner) -> None:
        __tracebackhide__ = True
        if hasattr(obj, "simulator"):
            raise RuntimeError(f"{obj}: function only available prior to .install()")

    def __init__(
        self,
        ts: float,
        te: float,
        *,
        accuracy: int = 1000000,
        need_synchronized: bool | None = None,
        install_to: Iterable[SimulatorInstallable] = [],
    ):
        """
        Args:
            ts: Simulation start time in seconds.
            te: Simulation end time in seconds, infinite means continuous simulation.
            accuracy: Number of time slots per second, defaults to 1000000 i.e. 1us time slot.
            need_synchronized: True to use thread-safe event pool, False to use non-thread-safe event pool,
                               default is thread-safe for continuous simulation and non-thread-safe for finite simulation.
            install_to: Install this simulator by invoking ``.install(self)`` on each target.
        """
        self.accuracy = accuracy

        assert ts >= 0.0
        self.ts: Final = self.time(sec=ts)
        """Simulation start time."""
        assert te >= ts
        self.te: Final = Time.SENTINEL if math.isinf(te) else self.time(sec=te)
        """Simulation end time. ``Time.SENTINEL`` means continuous simulation."""
        self.time_spend = 0.0
        """Wall-clock time for entire simulation run."""

        if (need_synchronized is None and self.is_continuous) or need_synchronized:
            pool_typ = SynchronizedEventPool
        else:
            pool_typ = HeapEventPool

        self._pool = pool_typ(self.ts.slot, None if self.is_continuous else self.te.slot)
        self.total_events = 0
        """How many events have been inserted into the simulator."""

        for install_target in install_to:
            install_target.install(self)

    @property
    def is_continuous(self) -> bool:
        """
        Determine whether the simulation is continuous (instead of finite).
        """
        return self.te is Time.SENTINEL

    @property
    def tc(self) -> Time:
        """
        Current simulation time.

        * When the simulation has not started, this is same as ``.ts``.
        * When a finite simulation has finished, this is same as ``.te``.
        * Otherwise, this reflects the time of the currently processing or last processed event.
        """
        return Time(self._pool.tc, accuracy=self.accuracy)

    @property
    def running(self) -> bool:
        """Is the simulator running?"""
        return self._pool.running

    @overload
    def time(self, *, slot: int) -> Time:
        """Produce ``Time`` from time slot."""

    @overload
    def time(self, *, sec: float) -> Time:
        """Produce ``Time`` from seconds."""

    def time(self, *, slot: int | None = None, sec: float = math.nan) -> Time:
        if slot is not None:
            return Time(slot, accuracy=self.accuracy)
        return Time.from_sec(sec, accuracy=self.accuracy)

    def sched(self, event: Event) -> None:
        """
        Add an event into simulator event pool.
        """
        assert event.t.accuracy == self.accuracy

        t = event.t.slot
        if t < self._pool.tc:
            log.warning("Event %s dropped: scheduled for %s but simulator is at %s", event, event.t, self.tc)
            return
        if self._pool.te is not None and t > self._pool.te:
            return

        self._pool.insert(event)
        self.total_events += 1

    def run(self) -> None:
        """
        Run the simulation.

        If ``MQNS_PROFILING=1`` environment variable is set, the simulation runs under cProfile profiling,
        and a profiling report is printed at the end of simulation.
        """
        __tracebackhide__ = self._run_tracebackhide

        profile = Profile() if os.getenv("MQNS_PROFILING", "0") == "1" else None
        log.info(
            f"{'Continuous' if self.is_continuous else 'Finite'} simulation started in {self._pool}"
            + (" with profiling." if profile else ".")
        )

        self._pool.start()
        trs = time.time()

        self._current_event: Event | None = None
        try:
            if profile:
                profile.runcall(self._run)
            else:
                self._run()
        except Exception as e:
            log.error("Simulator exception %s occurred during %s: %s", type(e).__name__, self._current_event, e)
            raise RuntimeError(f"simulation aborted at [{self.tc}] by exception: {e}") from e
        finally:
            self.stop()  # ensure s.running is False
            del self._current_event

        tre = time.time()
        self.time_spend = tre - trs
        sim_time = (self.tc - self.ts).sec
        log.info(
            f"{'Continuous' if self.is_continuous else 'Finite'} simulation finished, "
            f"runtime {self.time_spend}, {self.total_events} events, "
            f"sim_time {sim_time}, x{'INF' if self.time_spend == 0 else sim_time / self.time_spend}"
        )

        if profile:
            profile.print_stats(SortKey.TIME)

    def _run(self) -> None:
        __tracebackhide__ = self._run_tracebackhide

        while self._pool.running:
            self._current_event = event = self._pool.pop()

            if event is None:
                # simulator stopped or finite end time reached
                break

            if event.is_canceled:
                continue

            event.invoke()

            if self.watchers is not None and (monitors := self.watchers.get(event.__class__)) is not None:
                for monitor in monitors:
                    monitor.handle(event)

    def stop(self) -> None:
        """
        Stop the simulation loop.
        """
        self._pool.stop()

    @overload
    def update_gate(self, gate: Time, *, direct: Literal[True]) -> None: ...

    @overload
    def update_gate(self, gate: Time, *, priority=0xFFFFFFFF) -> Event: ...

    def update_gate(self, gate: Time, *, direct=False, priority=0xFFFFFFFF) -> Any:
        """
        Update the gate time to support external synchronization.
        Simulation is paused at this time slot until it's advanced.

        Args:
            gate: New gate time.
            direct: If True, update directly without scheduling an event.
                    This option should not be used from a secondary thread.
            priority: Event priority, defaults to 2^32-1 that represents a low priority.

        Returns:
            ``None`` if ``direct=True``.
            Otherwise, the gate update event.
            The secondary thread must cancel previously scheduled gate update events.

        Calling this function is mandatory when using thread-safe event pool.
        It is no-op when using non-thread-safe event pool.

        When the external program has released the gate, it should not schedule new events
        before the gate time, but may schedule new events at or after gate time.
        This restriction does not apply to internally scheduled events.
        """
        if direct:
            self._update_gate(gate)
            return

        event = func_to_event(self.tc, self._update_gate, gate)
        event.name = "Simulator.update_gate"
        event.priority = priority
        self.sched(event)
        return event

    def _update_gate(self, gate: Time) -> None:
        assert gate.accuracy == self.accuracy
        log.debug("Simulator.update_gate(%s)", gate.slot)
        self._pool.update_gate(gate.slot)

    def set_gate_reached_handler(self, h: Callable[[int], None]) -> None:
        """
        Set a callback function when the gate time has been reached.
        """
        self._pool.set_gate_reached_handler(h)
