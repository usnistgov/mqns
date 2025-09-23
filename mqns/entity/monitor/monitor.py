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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd
from typing_extensions import override

from mqns.entity.entity import Entity
from mqns.simulator import Event, Simulator, Time

if TYPE_CHECKING:
    from mqns.network.network import QuantumNetwork


AttributionFunc = Callable[[Simulator, "QuantumNetwork|None", Event], Any]
"""Callback function to calculate an attribution."""


class MonitorEvent(Event):
    """the event that notify the monitor to write down network status"""

    def __init__(self, t: Time, monitor: "Monitor", *, period: Time | None = None, name: str | None = None, by: Any = None):
        super().__init__(t, name, by)
        self.monitor = monitor
        self.period = period

    @override
    def invoke(self) -> None:
        self.monitor.handle(self)


class Monitor(Entity):
    def __init__(self, name: str, network: "QuantumNetwork|None" = None) -> None:
        """
        Virtual entity that helps users collect network status.

        Args:
            name: monitor name.
            network: the quantum network.

        """
        super().__init__(name=name)
        self.network = network
        self.attributions: list[tuple[str, AttributionFunc]] = []
        self.records: dict[str, list[Any]] = {"time": []}

        self.watch_at_start = False
        self.watch_at_finish = False
        self.watch_period: list[float] = []
        self.watch_event: list[type[Event]] = []

    def install(self, simulator: Simulator) -> None:
        super().install(simulator)

        if self.watch_at_start:
            simulator.add_event(MonitorEvent(simulator.ts, self, name="start watch event", by=self))

        if simulator.te is not None and self.watch_at_finish:
            simulator.add_event(MonitorEvent(simulator.te, self, name="finish watch event", by=self))

        for p in self.watch_period:
            simulator.add_event(
                MonitorEvent(simulator.ts, self, period=simulator.time(sec=p), name=f"period watch event({p})", by=self)
            )

        for event_type in self.watch_event:
            simulator.watch_event[event_type].append(self)

    @override
    def handle(self, event: Event) -> None:
        simulator = self.simulator

        self.records["time"].append(simulator.tc.sec)
        for name, calculate_func in self.attributions:
            self.records[name].append(calculate_func(simulator, self.network, event))

        if isinstance(event, MonitorEvent) and event.period is not None:
            event.t += event.period
            simulator.add_event(event)

    def get_data(self) -> pd.DataFrame:
        """
        Retrieve the collected data.

        """
        return pd.DataFrame(self.records)

    def add_attribution(self, name: str, calculate_func: AttributionFunc) -> None:
        """
        Set an attribution that will be recorded.
        For example, an attribution could be the throughput or the fidelity.

        Args:
            name: column name, e.g., fidelity, throughput, time.
            calculate_func: a function to calculate the value.

        Usage:
            m = Monitor()

            # record the event happening time
            m.add_attribution("time", lambda s,n,e: e.t)

            # get the 'name' attribution of the last node
            m.add_attribution("count", lambda s,network,e: network.nodes[-1].name)

        """
        assert self._simulator is None
        self.attributions.append((name, calculate_func))
        self.records[name] = []

    def at_start(self) -> None:
        """
        Watch the initial status before the simulation starts.
        """
        assert self._simulator is None
        self.watch_at_start = True

    def at_finish(self) -> None:
        """
        Watch the final status after the simulation.

        This does not work in a continuous simulation or if the simulation is stopped with `Simulator.stop()`.
        """
        assert self._simulator is None
        self.watch_at_finish = True

    def at_period(self, period_time: float) -> None:
        """
        Watch network status at a constant interval.

        Args:
            period_time (float): the interval in seconds.

        Usage:
            # record network status every 3 seconds.
            m.at_period(3)

        """
        assert self._simulator is None
        assert period_time > 0
        self.watch_period.append(period_time)

    def at_event(self, event_type: type[Event]) -> None:
        """
        Watch network status whenever the event happens.

        Args:
            event_type (Event): the watched event.

        Usage:
            # record network status when a node receives a qubit
            m.at_event(RecvQubitPacket)

        """
        assert self._simulator is None
        self.watch_event.append(event_type)
