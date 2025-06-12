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

import time
from typing import Optional

from qns.simulator.event import Event
from qns.simulator.pool import DefaultEventPool
from qns.simulator.ts import Time, default_accuracy
from qns.utils import log

from . import ts

default_start_second = 0.0


class Emulator:
    """The discrete-event driven simulator core, continuous mode enabled"""

    def __init__(self, start_second: float = default_start_second, accuracy: int = default_accuracy):
        """Args:
        start_second (float): the start second of the simulation
        accuracy (int): the number of time slots per second

        """
        self.accuracy = accuracy
        ts.default_accuracy = accuracy

        self.ts: Time = self.time(sec=start_second)
        self.time_spend: float = 0

        self.event_pool = DefaultEventPool(self.ts)
        self.status = {}
        self.total_events = 0

        self.watch_event = {}
        self._running = False

    @property
    def current_time(self) -> Time:
        """Get the current time of the simulation

        Returns:
            (Time) the current time

        """
        return self.event_pool.current_time

    @property
    def tc(self) -> Time:
        """The alias of `current_time`"""
        return self.current_time

    def time(self, time_slot: Optional[int] = None, sec: Optional[float] = None) -> Time:
        """Produce a ``Time`` using ``time_slot`` or ``sec``

        Args:
            time_slot (Optional[int]): the time slot
            sec (Optional[float]): the second
        Returns:
            the produced ``Time`` object

        """
        if time_slot is not None:
            return Time(time_slot=time_slot, accuracy=self.accuracy)
        return Time(sec=sec, accuracy=self.accuracy)

    def add_event(self, event: Event) -> None:
        """Add an ``event`` into simulator event pool.
        :param event: the inserting event
        """
        if self.event_pool.add_event(event):
            self.total_events += 1

    def stop(self) -> None:
        """Stop the continuous simulation loop"""
        self._running = False

    def run(self) -> None:
        """Run the simulation continuously until stopped."""
        log.debug("continuous simulation started.")

        self._running = True
        trs = time.time()

        while self._running:
            event = self.event_pool.next_event()

            if event is not None:
                if not event.is_canceled:
                    event.invoke()
                    monitor_list = self.watch_event.get(event.__class__, [])
                    for m in monitor_list:
                        m.handle(event)
            else:
                # No event available; idle briefly to wait for external events
                time.sleep(0.001)

        tre = time.time()
        self.time_spend = tre - trs
        log.debug("continuous simulation stopped.")

        if tre - trs == 0:
            log.debug(f"runtime {tre - trs}, {self.total_events} events, xINF")
        else:
            log.debug(f"runtime {tre - trs}, {self.total_events} events, x{self.current_time.sec / (tre - trs)}")
