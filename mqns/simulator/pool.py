import heapq
import threading
import time
from abc import ABC, abstractmethod
from typing import override

from mqns.simulator.event import Event


class EventPool(ABC):
    """
    EventPool stores events scheduled through the simulator and returns them in timeline order.
    """

    def __init__(self, ts: int, te: int | None):
        """
        Args:
            ts: Start time slot.
            te: End time slot, None means continuous.
        """
        self.running = False
        """Whether the simulator is running."""
        self.tc = ts
        """Current time slot."""
        self.te = te
        """End time slot, None means continuous."""
        self._list: list[Event] = []

    def start(self) -> None:
        """Set ``running = True``."""
        self.running = True

    def stop(self) -> None:
        """Set ``running = False``."""
        self.running = False

    def update_gate(self, t: int) -> None:
        """
        Update the gate time to support external synchronization.
        Simulation is paused at this time slot until it's advanced.
        """
        _ = t

    @abstractmethod
    def insert(self, event: Event) -> None:
        """
        Insert an event.

        Args:
            event: The event; its time accuracy must be consistent.
        """

    @abstractmethod
    def pop(self) -> Event | None:
        """
        Pop the next event to be executed.

        Returns:
            The next event, or None if there are no more events.
        """


class HeapEventPool(EventPool):
    """
    Heap-based event pool (non thread safe).
    """

    @override
    def insert(self, event: Event) -> None:
        heapq.heappush(self._list, event)

    @override
    def pop(self) -> Event | None:
        if not self._list:
            if self.te is None:
                time.sleep(0.001)  # idle briefly to wait for external events
            else:
                self.tc = self.te
            return None

        event = heapq.heappop(self._list)
        self.tc = event.t.time_slot
        return event

    def __repr__(self):
        return "<HeapEventPool>"


class SynchronizedEventPool(EventPool):
    """
    Synchronized event pool (thread safe).
    """

    def __init__(self, ts: int, te: int | None):
        super().__init__(ts, te)
        self._cv = threading.Condition()
        self._gate = ts

    @override
    def stop(self) -> None:
        super().stop()
        with self._cv:
            self._cv.notify_all()  # wake up .pop() so it can return

    @override
    def update_gate(self, t: int) -> None:
        assert self._gate <= t
        with self._cv:
            self._gate = t
            self._cv.notify_all()  # wake up .pop() if it's waiting at the gate

    @override
    def insert(self, event: Event) -> None:
        with self._cv:
            heapq.heappush(self._list, event)
            self._cv.notify_all()  # wake up .pop() if it's waiting for events

    @override
    def pop(self) -> Event | None:
        while True:
            with self._cv:
                if not self.running:  # simulator stopped
                    return None

                if not self._list and self.te is not None:  # finite simulation finished
                    self.tc = self.te
                    return None

                if self._list:  # has events
                    next_t = self._list[0].t.time_slot

                    if next_t <= self._gate:  # event is before gate and can be executed
                        event = heapq.heappop(self._list)
                        self.tc = next_t
                        return event

                # no event or event is after gate
                self._cv.wait(timeout=1)

            # release conditional variable so that the thread can be interrupted

    def __repr__(self):
        return "<SynchronizedEventPool>"
