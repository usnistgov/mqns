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

import heapq
import math

from mqns.simulator.event import Event


class DefaultEventPool:
    """
    Heap-based event pool.
    """

    def __init__(self, ts: int, te: int | None):
        """
        Constructor.

        Args:
            ts: Start time slot.
            te: End time slot.
        """
        self.tc = ts
        self.te = math.inf if te is None else te
        self.event_list: list[Event] = []

    def add_event(self, event: Event) -> bool:
        """
        Add an event.

        Args:
            event: The event; its time accuracy must be consistent.

        Returns:
            Whether the event has been inserted.
        """
        if event.t.time_slot < self.tc or event.t.time_slot > self.te:
            return False

        heapq.heappush(self.event_list, event)
        return True

    def next_event(self) -> Event | None:
        """
        Pop the next event to be executed.

        Returns:
            The next event, or None if the simulation has ended.
        """
        try:
            event = heapq.heappop(self.event_list)
            self.tc = event.t.time_slot
            return event
        except IndexError:
            if not math.isinf(self.te):
                self.tc = int(self.te)
            return None
