import time

import pytest

from mqns.utils import WallClockTimeout


@pytest.mark.parametrize(
    ("sleep_for", "occurred"),
    [
        (0.1, False),
        (0.7, True),
    ],
)
def test_timeout(sleep_for: float, occurred: bool):
    stopped = False

    def stop():
        nonlocal stopped
        stopped = True

    timeout = WallClockTimeout(0.4, stop)
    with timeout():
        time.sleep(sleep_for)

    assert timeout.occurred is occurred
    assert stopped is occurred
