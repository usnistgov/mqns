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
from collections.abc import Iterable
from pstats import SortKey
from typing import TYPE_CHECKING, Any, Protocol, overload

from mqns.simulator.event import Event
from mqns.simulator.pool import HeapEventPool
from mqns.simulator.time import Time
from mqns.utils import log

try:
    from cProfile import Profile
except ImportError:
    from profile import Profile

if TYPE_CHECKING:
    from mqns.entity.monitor import Monitor


class SimulatorInstallable(Protocol):
    def install(self, simulator: "Simulator") -> Any: ...


class Simulator:
    """
    Discrete-event driven simulator core.
    """

    watchers: dict[type[Event], list["Monitor"]] | None = None

    def __init__(
        self,
        start_second: float = 0.0,
        end_second: float = 60.0,
        *,
        accuracy: int = 1000000,
        install_to: Iterable[SimulatorInstallable] = [],
    ):
        """
        Args:
            start_second: simulation start time in seconds, defaults to 0.0.
            end_second: simulator end time in seconds, defaults to 60.0; infinite means continuous simulation.
            accuracy: the number of time slots per second, defaults to 1000000 i.e. 1us time slot.
            install_to: install this simulator by invoking `.install(self)` on each target.
        """
        self.accuracy = accuracy

        assert start_second >= 0.0
        self.ts = self.time(sec=start_second)
        """Simulation start time."""
        assert end_second >= start_second
        self.te = None if math.isinf(end_second) else self.time(sec=end_second)
        """Simulation end time. None means continuous simulation."""
        self.time_spend: float = 0
        """Wall-clock time for entire simulation run."""

        self.pool = HeapEventPool(self.ts.time_slot, None if self.te is None else self.te.time_slot)
        self.total_events = 0

        self._running = False

        for install_target in install_to:
            install_target.install(self)

    @property
    def tc(self) -> Time:
        """Current simulation time."""
        return self.time(time_slot=self.pool.tc)

    @property
    def running(self) -> bool:
        """Is the simulator running?"""
        return self._running

    @overload
    def time(self, *, time_slot: int) -> Time:
        """Produce ``Time`` from time slot."""

    @overload
    def time(self, *, sec: float) -> Time:
        """Produce ``Time`` from seconds."""

    def time(self, *, time_slot: int | None = None, sec: float = math.nan) -> Time:
        if time_slot is not None:
            return Time(time_slot, accuracy=self.accuracy)
        return Time.from_sec(sec, accuracy=self.accuracy)

    def add_event(self, event: Event) -> None:
        """
        Add an event into simulator event pool.
        """
        assert event.t.accuracy == self.accuracy
        if self.pool.insert(event):
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
        while self._running:
            event = self.pool.pop()

            if event is not None:
                if event.is_canceled:
                    continue

                event.invoke()

                if self.watchers is not None and (monitors := self.watchers.get(event.__class__)) is not None:
                    for monitor in monitors:
                        monitor.handle(event)

            elif self.te is None:  # continuous mode but no event available
                time.sleep(0.001)  # idle briefly to wait for external events

            else:  # finite simulation is finished
                self._running = False
                break

    def stop(self) -> None:
        """
        Stop the simulation loop.
        """
        self._running = False
