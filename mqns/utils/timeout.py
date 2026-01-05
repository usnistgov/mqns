import threading
from collections.abc import Callable
from contextlib import contextmanager


class WallClockTimeout:
    """
    Helper to enforce wall-clock timeout of the simulator run.
    """

    def __init__(self, timeout: float, stop: Callable[[], None]):
        """
        Args:
            timeout: wall-clock timeout in seconds.
            stop: function to stop the simulator, e.g. `simulator.stop()`.
        """
        self.occurred = False
        """
        Whether a timeout has occurred.
        """
        self._limit = timeout
        self._stop = stop
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._stop_after_timeout, daemon=True)

    def _stop_after_timeout(self):
        if not self._cancel.wait(timeout=self._limit):
            self._stop()
            self.occurred = True

    @contextmanager
    def __call__(self):
        """Open a context in which to run the simulation"""
        try:
            self._thread.start()
            yield None
        finally:
            self._cancel.set()
            self._thread.join()
