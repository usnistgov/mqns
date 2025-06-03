import logging

from qns.simulator.event import Event
from qns.simulator.simulator import Simulator
from qns.utils import log

log.logger.setLevel(logging.DEBUG)


class TimerEvent(Event):
    def invoke(self) -> None:
        log.info(f"{self.name}: it is {self.t} seconds")

    def __repr__(self) -> str:
        return f"<{self.name}-{self.t}>"


def test_simulator_with_log():
    s = Simulator(0, 15, 1000)

    e = TimerEvent(t=None, name="t0")
    s.add_event(e)
    # t0 is not scheduled because it does not have a timestamp
    assert s.total_events == 0

    t = 0
    while t <= 12:
        e = TimerEvent(t=s.time(sec=t), name="t1")
        s.add_event(e)
        t += 0.5
    # 25 instances of t1 scheduled at 0.0, 0.5, 1.0, .., 11.5, 12.0
    assert s.total_events == 25

    t = 5
    while t <= 20:
        e = TimerEvent(t=s.time(sec=t), name="t2")
        s.add_event(e)
        t += 1
    # 11 instances of t2 scheduled at 5, 6, .., 14, 15
    assert s.total_events == 25 + 11

    log.install(s)
    s.run()
