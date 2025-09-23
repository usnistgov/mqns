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


default_accuracy = 1000000  # {default_accuracy} time slots per second


def set_default_accuracy(time_slots: int):
    """Set the default simulation accuracy

    Args:
        time_slots (int): the time slots per second.

    """
    global default_accuracy
    default_accuracy = time_slots


def _to_time_slot(sec: int | float, accuracy: int) -> int:
    return round(sec * accuracy)


class Time:
    def __init__(self, time_slot: int = 0, sec: int | float = 0, accuracy: int | None = None):
        """Time: the time slot used in the simulator

        Args:
            time_slot (int): the time slot
            sec (float): the timestamp in second
            accuracy: time slots per second

        """
        self.accuracy = default_accuracy if accuracy is None else accuracy
        if time_slot != 0:
            self.time_slot = time_slot
        else:
            self.time_slot = _to_time_slot(sec, self.accuracy)

    @property
    def sec(self) -> float:
        """Returns:
        the timestamp in second

        """
        return self.time_slot / self.accuracy

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison operator.

        Time instance is equal to the other object only if:
        * The other object is also a Time instance.
        * They have the same accuracy.
        * They have the same time slot.
        """
        return isinstance(other, Time) and self.accuracy == other.accuracy and self.time_slot == other.time_slot

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __lt__(self, other: "Time") -> bool:
        """
        Less than comparison operator.
        Two Time instances can be compared only if they have the same accuracy.
        """
        assert self.accuracy == other.accuracy
        return self.time_slot < other.time_slot

    def __le__(self, other: "Time") -> bool:
        assert self.accuracy == other.accuracy
        return self.time_slot <= other.time_slot

    def __gt__(self, other: "Time") -> bool:
        assert self.accuracy == other.accuracy
        return self.time_slot > other.time_slot

    def __ge__(self, other: "Time") -> bool:
        assert self.accuracy == other.accuracy
        return self.time_slot >= other.time_slot

    def __hash__(self) -> int:
        return hash(self.time_slot)

    def __add__(self, ts: "Time|int|float") -> "Time":
        """
        Add a duration and returns a new Time object.

        Args:
            ts: either a Time object with same accuracy, or a duration number in seconds.
        """
        if isinstance(ts, Time):
            assert ts.accuracy == self.accuracy
            time_slot = ts.time_slot
        else:
            time_slot = _to_time_slot(ts, self.accuracy)
        return Time(time_slot=self.time_slot + time_slot, accuracy=self.accuracy)

    def __sub__(self, ts: "Time|int|float") -> "Time":
        """
        Subtract a duration and returns a new Time object.

        Args:
            ts: either a Time object with same accuracy, or a duration number in seconds.
        """
        if isinstance(ts, Time):
            assert ts.accuracy == self.accuracy
            time_slot = ts.time_slot
        else:
            time_slot = _to_time_slot(ts, self.accuracy)
        return Time(time_slot=self.time_slot - time_slot, accuracy=self.accuracy)

    def __repr__(self) -> str:
        return str(self.sec)
