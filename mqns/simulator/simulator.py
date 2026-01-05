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

import math
import os
import time
from collections import defaultdict
from pstats import SortKey
from typing import TYPE_CHECKING, overload

from mqns.simulator.event import Event
from mqns.simulator.pool import DefaultEventPool
from mqns.simulator.time import Time, default_accuracy, set_default_accuracy
from mqns.utils import log

try:
    from cProfile import Profile
except ImportError:
    from profile import Profile

if TYPE_CHECKING:
    from mqns.entity.monitor import Monitor

default_start_second = 0.0
default_end_second = 60.0


class Simulator:
    """
    Discrete-event driven simulator core.
    """

    def __init__(
        self,
        start_second: float = default_start_second,
        end_second: float = default_end_second,
        *,
        accuracy: int = default_accuracy,
    ):
        """
        Args:
            start_second: simulation start time in seconds.
            end_second: simulator end time in seconds; infinite means continuous simulation.
            accuracy: the number of time slots per second.
        """
        self.accuracy = accuracy
        set_default_accuracy(accuracy)

        assert start_second >= 0.0
        self.ts = self.time(sec=start_second)
        """Simulation start time."""
        assert end_second >= start_second
        self.te = None if math.isinf(end_second) else self.time(sec=end_second)
        """Simulation end time. None means continuous simulation."""
        self.time_spend: float = 0
        """Wall-clock time for entire simulation run."""

        self.event_pool = DefaultEventPool(self.ts.time_slot, None if self.te is None else self.te.time_slot)
        self.total_events = 0

        self.watch_event = defaultdict[type[Event], list["Monitor"]](lambda: [])

        self._running = False

    @property
    def tc(self) -> Time:
        """Current simulation time."""
        return self.time(time_slot=self.event_pool.tc)

    @property
    def running(self) -> bool:
        """Is the simulator running?"""
        return self._running

    @overload
    def time(self, *, time_slot: int) -> Time:
        """Produce `Time` from time slot."""
        pass

    @overload
    def time(self, *, sec: float) -> Time:
        """Produce `Time` from seconds."""
        pass

    def time(self, *, time_slot: int | None = None, sec: float = math.nan) -> Time:
        if time_slot is not None:
            return Time(time_slot, accuracy=self.accuracy)
        return Time.from_sec(sec, accuracy=self.accuracy)

    def add_event(self, event: Event) -> None:
        """
        Add an event into simulator event pool.
        """
        assert event.t.accuracy == self.accuracy
        if self.event_pool.add_event(event):
            self.total_events += 1

    def run(self) -> None:
        """
        Run the simulation.

        If MQNS_PROFILING=1 environment variable is set, the simulation runs under cProfile profiling,
        and a profiling report is printed at the end of simulation.
        """
        is_continuous = self.te is None
        profile = Profile() if os.getenv("MQNS_PROFILING", "0") == "1" else None
        log.info(f"{'Continuous' if is_continuous else 'Finite'} simulation started{' with profiling' if profile else ''}.")

        self._running = True
        trs = time.time()

        if profile:
            profile.runcall(self._run)
        else:
            self._run()

        tre = time.time()
        self.time_spend = tre - trs
        sim_time = (self.tc - self.ts).sec
        log.info(
            f"{'Continuous' if is_continuous else 'Finite'} simulation finished, "
            f"runtime {self.time_spend}, {self.total_events} events, "
            f"sim_time {sim_time}, x{'INF' if self.time_spend == 0 else sim_time / self.time_spend}"
        )

        if profile:
            profile.print_stats(SortKey.TIME)

    def _run(self) -> None:
        is_continuous = self.te is None
        while self._running:
            event = self.event_pool.next_event()
            if event is not None:
                if event.is_canceled:
                    continue
                event.invoke()
                for monitor in self.watch_event.get(event.__class__, []):
                    monitor.handle(event)
            elif is_continuous:
                time.sleep(0.001)  # idle briefly to wait for external events
            else:  # all events completed
                self._running = False
                break

    def stop(self) -> None:
        """
        Stop the simulation loop.
        """
        self._running = False
