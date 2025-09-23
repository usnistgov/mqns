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

from mqns.simulator.event import Event
from mqns.simulator.ts import Time


class DefaultEventPool:
    """The default implement of the event pool"""

    def __init__(self, ts: Time, te: Time | None):
        self.ts = ts
        """Event list start time."""
        self.te = te
        """Event list end time. None means infinity."""
        self.tc = ts
        """Current time."""
        self.event_list: list[Event] = []

    def add_event(self, event: Event) -> bool:
        """Insert an event into the pool

        Args:
            event (Event): The inserting event
        Returns:
            if the event is inserted successfully

        """
        if event.t < self.tc or (self.te is not None and event.t > self.te):
            return False

        heapq.heappush(self.event_list, event)
        return True

    def next_event(self) -> Event | None:
        """Get the next event to be executed

        Returns:
            The next event to be executed

        """
        try:
            event = heapq.heappop(self.event_list)
            self.tc = event.t
            return event
        except IndexError:
            if self.te is not None:
                self.tc = self.te
            return None
