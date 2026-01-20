import os
import sys
from logging import Logger, LoggerAdapter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Literal, cast, override

if TYPE_CHECKING:
    from mqns.simulator import Simulator


class CustomAdapter(LoggerAdapter):
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self._simulator: "Simulator|None" = None

    @override
    def process(self, msg, kwargs):
        if self._simulator:
            msg = f"[{self._simulator.tc}] {msg}"
        return msg, kwargs

    def install(self, simulator: "Simulator|None"):
        """
        Prepend simulator timestamp to log entries.
        """
        self._simulator = simulator

    def set_default_level(self, dflt_level: Literal["CRITICAL", "FATAL", "ERROR", "WARN", "INFO", "DEBUG"]):
        """
        Configure logging level.

        If `MQNS_LOGLVL` environment variable contains a valid log level, it is used.
        Otherwise, `dflt_level` is used as the logging level.
        """
        try:
            env_level = os.getenv("MQNS_LOGLVL", dflt_level)
            self.setLevel(env_level)
        except ValueError:  # MQNS_LOGLVL is not a valid level
            self.setLevel(dflt_level)


log = CustomAdapter(getLogger("mqns"))
"""
The default ``logger`` used by MQNS.
"""

log.set_default_level("INFO")
cast(Logger, log.logger).addHandler(StreamHandler(sys.stdout))
