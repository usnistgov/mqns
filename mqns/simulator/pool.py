import heapq
from abc import ABC, abstractmethod
from typing import override

from mqns.simulator.event import Event


class EventPool(ABC):
    """
    Event pool.
    """

    def __init__(self, ts: int, te: int | None):
        """
        Args:
            ts: Start time slot.
            te: End time slot, None means continuous.
        """
        self.tc = ts
        """Current time slot."""
        self.te = te
        """End time slot, None means continuous."""
        self._list: list[Event] = []

    def insert(self, event: Event) -> bool:
        """
        Insert an event.

        Args:
            event: The event; its time accuracy must be consistent.

        Returns: Whether the event has been inserted.
        """
        t = event.t.time_slot
        if t < self.tc or (self.te is not None and t > self.te):
            return False

        self._insert(event)
        return True

    @abstractmethod
    def _insert(self, event: Event) -> None:
        pass

    @abstractmethod
    def pop(self) -> Event | None:
        """
        Pop the next event to be executed.

        Returns:
            The next event, or None if there are no more events.
        """


class HeapEventPool(EventPool):
    """
    Heap-based event pool.
    """

    @override
    def _insert(self, event: Event) -> None:
        heapq.heappush(self._list, event)

    @override
    def pop(self) -> Event | None:
        if not self._list:
            if self.te is not None:
                self.tc = self.te
            return None

        event = heapq.heappop(self._list)
        self.tc = event.t.time_slot
        return event
