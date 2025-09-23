import pytest

from mqns.entity.timer import Timer
from mqns.simulator import Simulator, Time


def test_timer():
    s = Simulator(0, 10, 1000)

    trigger_times: list[Time] = []

    def trigger_func():
        trigger_times.append(s.tc)

    t1 = Timer("t1", 0, 10, 0.5, trigger_func)
    t1.install(s)
    s.run()

    assert len(trigger_times) == 21
    assert trigger_times[10].sec == pytest.approx(5.0)
