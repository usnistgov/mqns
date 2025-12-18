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
