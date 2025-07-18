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
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from qns.simulator.event import Event
from qns.simulator.pool import DefaultEventPool
from qns.simulator.ts import Time, default_accuracy
from qns.utils import log

from . import ts

if TYPE_CHECKING:
    from qns.entity.monitor import Monitor

default_start_second = 0.0
default_end_second = 60.0


class Simulator:
    """The discrete-event driven simulator core"""

    def __init__(
        self,
        start_second: float = default_start_second,
        end_second: float = default_end_second,
        accuracy: int = default_accuracy,
    ):
        """
        Args:
            start_second: simulation start time in seconds.
            end_second: simulator end time in seconds; infinite means continuous simulation.
            accuracy: the number of time slots per second.
        """
        self.accuracy = accuracy
        ts.default_accuracy = accuracy

        assert start_second >= 0.0
        self.ts = self.time(sec=start_second)
        """Simulation start time."""
        assert end_second >= start_second
        self.te = None if math.isinf(end_second) else self.time(sec=end_second)
        """Simulation end time. None means continuous simulation."""
        self.time_spend: float = 0
        """Wallclock time for entire simulation run."""

        self.event_pool = DefaultEventPool(self.ts, self.te)
        self.status = {}
        self.total_events = 0

        self.watch_event = defaultdict[type[Event], list["Monitor"]](lambda: [])

        self._running = False

    @property
    def tc(self) -> Time:
        """Current simulator time."""
        return self.event_pool.tc

    def time(self, time_slot: int | None = None, sec: int | float | None = None) -> Time:
        """Produce a ``Time`` using either ``time_slot`` or ``sec``

        Args:
            time_slot (Optional[int]): the time slot
            sec (Optional[float]): the second
        Returns:
            the produced ``Time`` object

        """
        if time_slot is not None:
            return Time(time_slot=time_slot, accuracy=self.accuracy)
        assert sec is not None
        return Time(sec=sec, accuracy=self.accuracy)

    def add_event(self, event: Event) -> None:
        """Add an ``event`` into simulator event pool.
        :param event: the inserting event
        """
        if self.event_pool.add_event(event):
            self.total_events += 1

    def run(self) -> None:
        """
        Run the simulation.
        """
        is_continuous = self.te is None
        log.info(f"{'Continuous' if is_continuous else 'Finite'} simulation started.")

        self._running = True
        trs = time.time()

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

        tre = time.time()
        self.time_spend = tre - trs
        sim_time = (self.tc - self.ts).sec
        log.info(
            f"{'Continuous' if is_continuous else 'Finite'} simulation finished, "
            f"runtime {self.time_spend}, {self.total_events} events, "
            f"sim_time {sim_time}, x{'INF' if self.time_spend == 0 else sim_time / self.time_spend}"
        )

    def stop(self) -> None:
        """Stop the simulation loop"""
        self._running = False
