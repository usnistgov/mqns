import pytest

from qns.simulator.ts import Time, set_default_accuracy


class ChangeDefaultAccuracy:
    def __init__(self, accuracy: int):
        self.new_accuracy = accuracy
        self.old_accuracy: int

    def __enter__(self):
        self.old_accuracy = Time().accuracy
        set_default_accuracy(self.new_accuracy)

    def __exit__(self, exc_type, exc_value, traceback):
        set_default_accuracy(self.old_accuracy)


def test_time_compare():
    with ChangeDefaultAccuracy(1000000):
        t1 = Time(1)
        t2 = Time(sec=1.1)
        t3 = Time()
        t4 = Time(1100000)

    assert t1 == t1 # noqa: PLR0124
    assert t2 >= t1
    assert t1 <= t2
    assert t1 < t2
    assert t3 < t1
    assert t1 != t4
    assert t2 == t4

    t0 = Time(sec=1.1, accuracy=2000)
    assert t2 != t0
    with pytest.raises(AssertionError):
        _ = t3 < t0

    assert t2 != 1
    assert t2 != "A"


def test_time_accuracy():
    t0 = Time(sec=1.0)

    with ChangeDefaultAccuracy(2000):
        t1 = Time(sec=1.0)
        t2 = Time(sec=1.0, accuracy=3000)

    t3 = Time(sec=1.0)
    t4 = Time(sec=1.0, accuracy=4000)

    assert t0.sec == pytest.approx(1.0)
    assert t1.sec == pytest.approx(1.0)
    assert t2.sec == pytest.approx(1.0)
    assert t3.sec == pytest.approx(1.0)
    assert t4.sec == pytest.approx(1.0)

    assert t0.accuracy == t3.accuracy
    assert t1.accuracy == 2000
    assert t2.accuracy == 3000
    assert t4.accuracy == 4000


def test_time_add_sub():
    t5 = Time(sec=5, accuracy=1000)

    t6a = t5 + 1
    t6b = t5 + 1.0
    t6c = t5 + Time(sec=1.0, accuracy=1000)
    assert t6a.sec == pytest.approx(6.0)
    assert t6b.sec == pytest.approx(6.0)
    assert t6c.sec == pytest.approx(6.0)

    with pytest.raises(AssertionError):
        _ = t5 + Time(sec=1.0, accuracy=2000)

    t3a = t5 - 2
    t3b = t5 - 2.0
    t3c = t5 - Time(sec=2.0, accuracy=1000)
    assert t3a.sec == pytest.approx(3.0)
    assert t3b.sec == pytest.approx(3.0)
    assert t3c.sec == pytest.approx(3.0)

    with pytest.raises(AssertionError):
        _ = t5 - Time(sec=2.0, accuracy=2000)


def print_msg(msg):
    print(msg)


def test_simulator_time():
    """
    If we modify the default_accuracy of the simulator,
    check whether the accuracy of subsequent events will be automatically synchronized with the simulator
    without special modification.
    """
    from qns.simulator.event import func_to_event
    from qns.simulator.simulator import Simulator
    s = Simulator(1, 10, 1000)
    s.run()
    print_event = func_to_event(Time(sec=1), print_msg, "hello world")
    assert print_event.t is not None
    assert print_event.t.accuracy == 1000
