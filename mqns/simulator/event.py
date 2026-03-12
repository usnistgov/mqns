from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, override

from mqns.simulator.time import Time


class Event(ABC):
    """Event in simulator."""

    is_canceled: bool = False
    """
    Whether the event has been canceled.

    The class attribute must not be modified, but it may be overwritten at instance level.
    Use ``event.cancel()`` to cancel an event.
    """

    priority: int = 0
    """
    Event priority within same time slot.
    Events with smaller priority number are invoked before events with larger priority number.
    Events sharing same time slot and same priority number may be invoked in any order.

    The class attribute must not be modified, but it may be overwritten at instance level.
    """

    def __init__(self, t: Time, name: str | None = None):
        self.t = t
        self.name = name

    @abstractmethod
    def invoke(self) -> None:
        """Invoke the event."""

    def cancel(self) -> None:
        """Cancel the event."""
        self.is_canceled = True

    def __lt__(self, other: "Event") -> bool:
        """Compare event ordering in Simulator heap."""
        if self.t.time_slot != other.t.time_slot:
            return self.t.time_slot < other.t.time_slot
        return self.priority < other.priority

    def __repr__(self) -> str:
        return f"Event({self.name or ''})"


class WrapperEvent(Event):
    def __init__(self, t: Time, fn: Callable, args: Any, kwargs: Any):
        super().__init__(t)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @override
    def invoke(self) -> None:
        self.fn(*self.args, **self.kwargs)


def func_to_event(t: Time, fn: Callable, *args, **kwargs):
    """
    Convert a function to an event, the function ``fn`` will be called at ``t``.
    It is a simple method to wrap a function to an event.

    Args:
        t: timestamp to call the function.
        fn: the function.
        *args: the function's positional parameters.
        **kwargs: the function's keyword parameters.
    """
    return WrapperEvent(t, fn, args, kwargs)
