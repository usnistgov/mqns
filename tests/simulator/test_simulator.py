import math
from collections import defaultdict
from typing import cast, override

from mqns.simulator import Event, Simulator, Time


class SimpleEvent(Event):
    invoke_count = defaultdict[str, int](lambda: 0)

    @override
    def invoke(self) -> None:
        self.invoke_count[cast(str, self.name)] += 1


class StopEvent(SimpleEvent):
    def __init__(self, t: Time, name: str, simulator: Simulator):
        super().__init__(t, name)
        self.simulator = simulator

    def invoke(self) -> None:
        super().invoke()
        assert self.simulator.running
        self.simulator.stop()
        assert not self.simulator.running


def test_simulator_run():
    SimpleEvent.invoke_count.clear()
    simulator = Simulator(0, 15, 1000)
    assert simulator.total_events == 0

    e = SimpleEvent(simulator.time(sec=1), name="t0")
    simulator.add_event(e)
    e.cancel()
    # 1 instance of t0 scheduled at 1.0 but will not be invoked
    assert simulator.total_events == 1

    t = 0
    while t <= 12:
        e = SimpleEvent(simulator.time(sec=t), name="t1")
        simulator.add_event(e)
        t += 0.5
    # 25 instances of t1 scheduled at 0.0, 0.5, 1.0, .., 11.5, 12.0
    assert simulator.total_events == 1 + 25

    t = 5
    while t <= 20:
        e = SimpleEvent(simulator.time(sec=t), name="t2")
        simulator.add_event(e)
        t += 1
    # 11 instances of t2 scheduled at 5, 6, .., 14, 15
    assert simulator.total_events == 1 + 25 + 11

    assert not simulator.running
    simulator.run()
    assert simulator.tc == simulator.te
    assert not simulator.running

    assert SimpleEvent.invoke_count["t0"] == 0
    assert SimpleEvent.invoke_count["t1"] == 25
    assert SimpleEvent.invoke_count["t2"] == 11


def do_test_stop(*, te: float):
    SimpleEvent.invoke_count.clear()
    simulator = Simulator(0, te, 1000)

    e = StopEvent(simulator.time(sec=9.5), name="s0", simulator=simulator)
    simulator.add_event(e)
    # 1 instance of s0 scheduled at 9.5
    assert simulator.total_events == 1

    t = 1
    while t <= 60:
        e = SimpleEvent(simulator.time(sec=t), name="t1")
        simulator.add_event(e)
        t += 1
    # up to 60 instances of t2 scheduled at 1, 2, .., MIN(60, te_sec)
    assert simulator.total_events == 1 + min(60, te)

    simulator.run()
    assert simulator.tc.sec < te

    assert SimpleEvent.invoke_count["t1"] == 9
    assert SimpleEvent.invoke_count["s0"] == 1


def test_simulator_run_stop():
    do_test_stop(te=15)


def test_simulator_continuous_stop():
    do_test_stop(te=math.inf)
