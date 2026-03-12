import math
import threading
import time
from collections import defaultdict
from typing import override

import pytest

from mqns.simulator import Event, Simulator, Time


class SimpleEvent(Event):
    seq = 0
    invokes = defaultdict[str, list[int]](lambda: [])

    @override
    def invoke(self) -> None:
        self.invokes[self.name or ""].append(SimpleEvent.seq)
        SimpleEvent.seq += 1


class RescheduleEvent(SimpleEvent):
    def __init__(self, t: Time, name: str, simulator: Simulator, *new_events: tuple[str, int]):
        super().__init__(t, name)
        self.simulator = simulator
        self.new_events = new_events

    @override
    def invoke(self) -> None:
        super().invoke()

        for name, prio in self.new_events:
            event = SimpleEvent(self.t, name)
            event.priority = prio
            self.simulator.add_event(event)


class StopEvent(SimpleEvent):
    def __init__(self, t: Time, name: str, simulator: Simulator):
        super().__init__(t, name)
        self.simulator = simulator

    @override
    def invoke(self) -> None:
        super().invoke()
        assert self.simulator.running
        self.simulator.stop()
        assert not self.simulator.running


@pytest.fixture(autouse=True)
def clear_invokes():
    yield
    SimpleEvent.seq = 0
    SimpleEvent.invokes.clear()


def test_run():
    s = Simulator(0, 15, accuracy=1000)
    assert s.total_events == 0

    e = SimpleEvent(s.time(sec=1), name="t0")
    s.add_event(e)
    assert e.is_canceled is False
    e.cancel()
    assert e.is_canceled is True
    # 1 instance of t0 scheduled at 1.0 but will not be invoked
    assert s.total_events == 1

    t = 0
    while t <= 12:
        s.add_event(SimpleEvent(s.time(sec=t), name="t1"))
        t += 0.5
    # 25 instances of t1 scheduled at 0.0, 0.5, 1.0, .., 11.5, 12.0
    assert s.total_events == 1 + 25

    t = 5
    while t <= 20:
        s.add_event(SimpleEvent(s.time(sec=t), name="t2"))
        t += 1
    # 11 instances of t2 scheduled at 5, 6, .., 14, 15
    assert s.total_events == 1 + 25 + 11

    assert not s.running
    s.run()
    assert s.tc == s.te
    assert not s.running

    assert len(SimpleEvent.invokes["t0"]) == 0
    assert len(SimpleEvent.invokes["t1"]) == 25
    assert len(SimpleEvent.invokes["t2"]) == 11


def test_ordering():
    s = Simulator(0, 10, accuracy=1000)
    t1 = s.time(sec=1)
    t2 = s.time(sec=2)

    p19 = SimpleEvent(t1, "p19")
    p19.priority = 9
    s.add_event(p19)
    p11 = SimpleEvent(t1, "p11")
    p11.priority = 1
    s.add_event(p11)
    p15 = SimpleEvent(t1, "p15")
    p15.priority = 5
    s.add_event(p15)

    p25 = RescheduleEvent(t2, "p25", s, ("p21", 1), ("p29", 9))
    p25.priority = 5
    s.add_event(p25)

    s.run()

    assert SimpleEvent.invokes == {
        "p11": [0],
        "p15": [1],
        "p19": [2],
        "p21": [4],
        "p25": [3],
        "p29": [5],
    }


@pytest.mark.parametrize(
    "te",
    [
        15,
        math.inf,
    ],
)
def test_stop(*, te: float):
    s = Simulator(0, te, accuracy=1000)
    s.update_gate(s.time(sec=60), direct=True)

    e = StopEvent(s.time(sec=9.5), name="s0", simulator=s)
    s.add_event(e)
    # 1 instance of s0 scheduled at 9.5
    assert s.total_events == 1

    t = 1
    while t <= 60:
        e = SimpleEvent(s.time(sec=t), name="t1")
        s.add_event(e)
        t += 1
    # up to 60 instances of t1 scheduled at 1, 2, .., MIN(60, te)
    assert s.total_events == 1 + min(60, te)

    s.run()
    assert s.tc.sec < te

    assert len(SimpleEvent.invokes["t1"]) == 9
    assert len(SimpleEvent.invokes["s0"]) == 1


def test_gate():
    s = Simulator(0.1, math.inf, accuracy=1000)
    assert s.tc == s.ts == s.time(sec=0.1)
    s.add_event(SimpleEvent(s.time(sec=0), "z0"))  # before s.ts, dropped

    # set initial gate to 5s, schedule events
    s.update_gate(s.time(sec=5), direct=True)
    s.add_event(SimpleEvent(s.time(sec=2), "b2"))
    s.add_event(SimpleEvent(s.time(sec=5), "b5"))
    s.add_event(SimpleEvent(s.time(sec=8), "a8"))

    # run Simulator in a background thread
    th = threading.Thread(target=s.run, daemon=True)
    th.start()

    # let the Simulator hit the gate, verify b2,b5 invoked
    time.sleep(0.2)
    assert len(SimpleEvent.invokes["b2"]) == 1
    assert len(SimpleEvent.invokes["b5"]) == 1
    assert len(SimpleEvent.invokes["a8"]) == 0
    assert s.tc == s.time(sec=5)
    assert th.is_alive()

    # schedule another event at initial gate and release the gate to 10s
    s.add_event(SimpleEvent(s.time(sec=4), "z4"))  # before s.tc, dropped
    s.add_event(SimpleEvent(s.time(sec=5), "a5"))

    # let the Simulator run again, verify a5,a8 invoked
    update_gate_event = s.update_gate(s.time(sec=10))
    assert update_gate_event.t == s.time(sec=5)
    assert update_gate_event.priority > 0
    time.sleep(0.2)
    assert len(SimpleEvent.invokes["a5"]) == 1
    assert len(SimpleEvent.invokes["a8"]) == 1
    assert s.tc == s.time(sec=8)
    assert th.is_alive()

    # verify dropped events are not invoked
    assert len(SimpleEvent.invokes["z0"]) == 0
    assert len(SimpleEvent.invokes["z4"]) == 0
    assert s.total_events == 5  # b2, b5, Simulator.update_gate, a5, a8

    # stop and cleanup
    s.stop()
    th.join(timeout=1)
    assert not th.is_alive()
