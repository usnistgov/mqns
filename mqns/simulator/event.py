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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, override

from mqns.simulator.ts import Time


class Event(ABC):
    """Basic event class in simulator"""

    def __init__(self, t: Time, name: str | None = None, by: Any = None):
        """Args:
        t (Time): the time slot of this event
        by: the entity or application that causes this event
        name (str): the name of this event

        """
        self.t = t
        self.name = name
        self.by = by
        self._is_canceled: bool = False

    @abstractmethod
    def invoke(self) -> None:
        """Invoke the event."""
        pass

    def cancel(self) -> None:
        """Cancel this event"""
        self._is_canceled = True

    @property
    def is_canceled(self) -> bool:
        """Returns:
        whether this event has been canceled

        """
        return self._is_canceled

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Event) and self.t.time_slot == other.t.time_slot

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __lt__(self, other: "Event") -> bool:
        return self.t.time_slot < other.t.time_slot

    def __le__(self, other: "Event") -> bool:
        return self.t.time_slot <= other.t.time_slot

    def __gt__(self, other: "Event") -> bool:
        return not self <= other

    def __ge__(self, other: "Event") -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash(self.t)

    def __repr__(self) -> str:
        return f"Event({self.name or ''})"


class WrapperEvent(Event):
    def __init__(self, t: Time, name: str | None, by: Any, fn: Callable, args: Any, kwargs: Any):
        super().__init__(t, name, by)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @override
    def invoke(self) -> None:
        self.fn(*self.args, **self.kwargs)


def func_to_event(t: Time, fn: Callable, *args, name: str | None = None, by: Any = None, **kwargs):
    """Convert a function to an event, the function `fn` will be called at `t`.
    It is a simple method to wrap a function to an event.

    Args:
        t: the function will be called at `t`
        fn: the function
        *args: the function's positional parameters
        name: event name
        by: the entity or application that will causes this event
        **kwargs: the function's keyword parameters

    """
    return WrapperEvent(t, name, by, fn, args, kwargs)
