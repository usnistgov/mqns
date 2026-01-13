import pytest

from mqns.entity.timer import Timer
from mqns.simulator import Simulator, Time


def test_timer():
    trigger_times: list[Time] = []

    def trigger_func():
        trigger_times.append(s.tc)

    t1 = Timer("t1", 0, 10, 0.5, trigger_func)

    s = Simulator(0, 10, accuracy=1000, install_to=(t1,))
    s.run()

    assert len(trigger_times) == 21
    assert trigger_times[10].sec == pytest.approx(5.0)
