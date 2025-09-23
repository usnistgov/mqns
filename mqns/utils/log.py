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

import logging
import os
import sys
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mqns.simulator import Simulator

logger = logging.getLogger("mqns")
"""
The default ``logger`` used by SimQN
"""


def set_default_level(dflt_level: Literal["CRITICAL", "FATAL", "ERROR", "WARN", "INFO", "DEBUG"]):
    """
    Configure logging level.

    If `QNS_LOGLVL` environment variable contains a valid log level, it is used.
    Otherwise, `dflt_level` is used as the logging level.
    """
    try:
        env_level = os.getenv("QNS_LOGLVL", dflt_level)
        logger.setLevel(env_level)
    except ValueError:  # QNS_LOGLVL is not a valid level
        logger.setLevel(dflt_level)


def install(simulator: "Simulator"):
    """Install the logger to the simulator

    Args:
        s (Simulator): the simulator

    """
    logger._simulator = simulator


def debug(msg, *args):
    if hasattr(logger, "_simulator"):
        logger.debug(f"[{logger._simulator.tc}] " + msg, *args, stacklevel=2)
    else:
        logger.debug(msg, *args, stacklevel=2)


def info(msg, *args):
    if hasattr(logger, "_simulator"):
        logger.info(f"[{logger._simulator.tc}] " + msg, *args, stacklevel=2)
    else:
        logger.info(msg, *args, stacklevel=2)


def error(msg, *args):
    if hasattr(logger, "_simulator"):
        logger.error(f"[{logger._simulator.tc}] " + msg, *args, stacklevel=2)
    else:
        logger.error(msg, *args, stacklevel=2)


def warn(msg, *args):
    if hasattr(logger, "_simulator"):
        logger.warning(f"[{logger._simulator.tc}] " + msg, *args, stacklevel=2)
    else:
        logger.warning(msg, *args, stacklevel=2)


def critical(msg, *args):
    if hasattr(logger, "_simulator"):
        logger.critical(f"[{logger._simulator.tc}] " + msg, *args, stacklevel=2)
    else:
        logger.critical(msg, *args, stacklevel=2)


def monitor(*args, sep: str = ",", with_time: bool = False):
    attrs = list(args)
    if with_time:
        attrs.insert(0, logger._simulator.tc)
    attrs_s = [str(a) for a in attrs]
    msg = sep.join(attrs_s)
    logger.info(msg)


set_default_level("INFO")
handle = logging.StreamHandler(sys.stdout)
logger.addHandler(handle)
