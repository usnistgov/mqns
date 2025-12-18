#    Modified by Amar Abane for Multiverse Quantum Network Simulator
#    Date: 05/17/2025
#    Summary of changes: Adapted logic to support dynamic approaches.
#
#    This file is based on a snapshot of SimQN (https://github.com/QNLab-USTC/SimQN),
#    which is licensed under the GNU General Public License v3.0.
#
#    The original SimQN header is included below.


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

from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar, cast, override

from mqns.entity.entity import Entity
from mqns.entity.node.app import Application, ApplicationT
from mqns.simulator import Event, Simulator

if TYPE_CHECKING:
    from mqns.entity.base_channel import ChannelT
    from mqns.entity.cchannel import ClassicChannel
    from mqns.network.network import QuantumNetwork, TimingMode


class Node(Entity):
    """Node is a generic node in the quantum network"""

    def __init__(self, name: str, *, apps: list[Application] | None = None):
        """
        Args:
            name: node name
            apps: applications on the node.
        """
        super().__init__(name=name)
        self._network: "QuantumNetwork|None" = None
        self.cchannels: list["ClassicChannel"] = []
        self._cchannel_by_dst = dict["Node", "ClassicChannel"]()
        self.apps: list[Application] = [] if apps is None else apps
        self._app_by_type = dict[type, Application]()

    @override
    def install(self, simulator: Simulator) -> None:
        """Called from Network.install()"""
        super().install(simulator)
        # initiate sub-entities
        from mqns.entity.cchannel import ClassicChannel  # noqa: PLC0415

        self._install_channels(ClassicChannel, self.cchannels, self._cchannel_by_dst)

        # initiate applications
        apps_by_type = defaultdict[type, list[Application]](lambda: [])
        for app in self.apps:
            apps_by_type[type(app)].append(app)
            app.install(self, simulator)
        for typ, apps in apps_by_type.items():
            if len(apps) == 1:
                self._app_by_type[typ] = apps[0]

    @override
    def handle(self, event: Event) -> None:
        """
        Dispatch an `Event` that happens on this Node.
        This event is passed to every application in apps list in order.

        Args:
            event (Event): the event that happens on this Node

        """
        for app in self.apps:
            skip = app.handle(event)
            if skip:
                break

    def add_apps(self, app: Application | list[Application]):
        """
        Insert one or more applications into the app list.

        Args:
            app: an application or a list of applications.
                 The caller is responsible for `deepcopy` if needed, so that each node has a separate instance.

        """
        if isinstance(app, list):
            self.apps += app
        else:
            self.apps.append(app)

    def get_apps(self, app_type: type[ApplicationT]) -> list[ApplicationT]:
        """
        Retrieve applications of given type.

        Args:
            app_type: Application type/class.
        """
        return [app for app in self.apps if isinstance(app, app_type)]

    def get_app(self, app_type: type[ApplicationT]) -> ApplicationT:
        """
        Retrieve an application of given type.
        There must be exactly one instance of this application.

        Args:
            app_type: Application type/class.

        Raises:
            IndexError - application does not exist, or there are multiple instances
        """
        return cast(ApplicationT, self._app_by_type[app_type])

    def _add_channel(self, channel: "ChannelT", channels: list["ChannelT"]) -> None:
        assert self._simulator is None
        channel.node_list.append(self)
        channels.append(channel)

    def _install_channels(
        self, typ: type["ChannelT"], channels: list["ChannelT"], by_neighbor: dict["Node", "ChannelT"]
    ) -> None:
        simulator = self.simulator
        for ch in channels:
            assert isinstance(ch, typ)
            for dst in ch.node_list:
                if dst != self:
                    by_neighbor[dst] = ch
            ch.install(simulator)

    @staticmethod
    def _get_channel(dst: "Node", by_neighbor: dict["Node", "ChannelT"]) -> "ChannelT":
        return by_neighbor[dst]

    def add_cchannel(self, cchannel: "ClassicChannel"):
        """
        Add a classic channel in this Node.
        This function is available prior to calling .install().
        """
        self._add_channel(cchannel, self.cchannels)

    def get_cchannel(self, dst: "Node") -> "ClassicChannel":
        """
        Retrieve the classic channel that connects to `dst`.

        Raises:
            IndexError - channel does not exist
        """
        return self._get_channel(dst, self._cchannel_by_dst)

    def add_network(self, network: "QuantumNetwork"):
        """
        Assign a network object to this node.
        """
        self._network = network

    @property
    def network(self) -> "QuantumNetwork":
        """
        Return the QuantumNetwork that this node belongs to.

        Raises:
            IndexError - network does not exist
        """
        if self._network is None:
            raise IndexError(f"node {repr(self)} is not in a network")
        return self._network

    @property
    def timing(self) -> "TimingMode":
        """
        Access the network-wide application timing mode.
        """
        return self.network.timing

    def __repr__(self) -> str:
        return f"<node {self.name}>"


NodeT = TypeVar("NodeT", bound=Node)
"""Type argument for Node or its subclass."""
