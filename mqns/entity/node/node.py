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
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Final, cast, override

from mqns.entity.entity import Entity
from mqns.entity.node.app import Application
from mqns.simulator import Event, Simulator

if TYPE_CHECKING:
    from mqns.entity.base_channel import BaseChannel
    from mqns.entity.cchannel import ClassicChannel, ClassicPacket
    from mqns.network.network import QuantumNetwork, TimingMode


class Node(Entity):
    """Node is a generic node in the quantum network."""

    apps: Final[Sequence[Application]]
    """Applications on this node."""

    cchannels: Final[Sequence["ClassicChannel"]]
    """Classic channels connected to this node."""

    def __init__(self, name: str, *, apps: Iterable[Application] | None = None):
        """
        Args:
            name: Node name.
            apps: Applications on the node.
                  More applications can be added via :py:meth:`~mqns.entity.node.Node.add_apps`.
                  The caller is responsible for ``deepcopy`` if needed, so that each node has a separate instance.
        """
        super().__init__(name)
        self._apps = list(apps or [])
        self._app_by_type: dict[type, Application] | None = None
        self._cchannels: list[ClassicChannel] = []
        self._cchannel_by_dst: dict[Node, ClassicChannel] = {}

        self.apps = self._apps
        self.cchannels = self._cchannels

    @override
    def install(self, simulator: Simulator) -> None:
        """Attach ``Simulator`` to entities within node."""
        super().install(simulator)
        self._install_channels(self._cchannels, self._cchannel_by_dst)
        self._node_install()

        apps_by_type = defaultdict[type, list[Application]](list)
        for app in self.apps:
            apps_by_type[type(app)].append(app)
            app.install(self)

        self._app_by_type = {}
        for typ, apps in apps_by_type.items():
            if len(apps) == 1:
                self._app_by_type[typ] = apps[0]

    def _node_install(self) -> None:
        pass

    @override
    def handle(self, event: Event) -> None:
        """
        Dispatch an event to applications.

        Multiple applications may have handlers for the same event.
        They are called in the order as they appear in ``self.apps`` list.
        If an application cancel the event, subsequent applications will not be called.
        """
        for app in self.apps:
            if event.is_canceled:
                return
            app.handle(event)

    def add_apps(self, app: Application | Iterable[Application]):
        """
        Insert one or more applications into the app list.

        Args:
            app: An application or a list of applications.
                 The caller is responsible for ``deepcopy`` if needed, so that each node has a separate instance.

        """
        Simulator.ensure_not_installed_to(self)
        if isinstance(app, Application):
            self._apps.append(app)
        else:
            self._apps.extend(app)

    def get_apps[A: Application](self, app_type: type[A]) -> list[A]:
        """
        Retrieve applications of given type.

        Args:
            app_type: Application type/class.
        """
        return [app for app in self.apps if isinstance(app, app_type)]

    def get_app[A: Application](self, app_type: type[A]) -> A:
        """
        Retrieve an application of given type.
        There must be exactly one instance of this application.

        Args:
            app_type: Application type/class.

        Raises:
            LookupError: Application does not exist, or there are multiple instances.
        """
        if self._app_by_type is None:  # this is called before self.install() populates _app_by_type
            Simulator.ensure_not_installed_to(self)
            return self._get_app_from_apps(app_type)

        try:
            return cast(A, self._app_by_type[app_type])
        except KeyError:
            # app_type is a base class
            return self._get_app_from_apps(app_type)

    def _get_app_from_apps[A: Application](self, app_type: type[A]) -> A:
        apps = self.get_apps(app_type)
        if len(apps) != 1:
            raise LookupError(f"{self} does not have exactly one instance of {app_type}")
        return apps[0]

    def _add_channel[C: "BaseChannel"](self, channel: C, channels: list[C]) -> None:
        Simulator.ensure_not_installed_to(self)
        channel.node_list.append(self)
        channels.append(channel)

    def _install_channels[C: "BaseChannel"](self, channels: list[C], by_neighbor: dict["Node", C]) -> None:
        for ch in channels:
            for dst in ch.node_list:
                if dst is not self:
                    by_neighbor[dst] = ch
            ch.install(self.simulator)

    @staticmethod
    def _get_channel[C: "BaseChannel"](dst: "Node", by_neighbor: dict["Node", C]) -> C:
        return by_neighbor[dst]

    def add_cchannel(self, cchannel: "ClassicChannel") -> None:
        """
        Add a classic channel in this node.

        This function is available prior to calling :py:meth:`~mqns.entity.node.Node.install`.
        """
        Simulator.ensure_not_installed_to(self)
        self._add_channel(cchannel, self._cchannels)

    def get_cchannel(self, dst: "Node") -> "ClassicChannel":
        """
        Retrieve the classic channel that connects to ``dst``.

        Raises:
            LookupError: Channel does not exist.
        """
        return self._get_channel(dst, self._cchannel_by_dst)

    def send_cpacket(self, next_hop: "Node", pkt: "ClassicPacket") -> None:
        """
        Send a classic packet to an adjacent node.

        Args:
            next_hop: An adjacent node that shares a classic channel with this node.
            pkt: The packet.
        """
        self.get_cchannel(next_hop).send(pkt, next_hop)

    def add_network(self, network: "QuantumNetwork"):
        """
        Assign a network object to this node.
        """
        self.network = network
        """Quantum network that contains this node."""

    @property
    def timing(self) -> "TimingMode":
        """
        Access the network-wide application timing mode.
        """
        return self.network.timing

    def __repr__(self) -> str:
        return f"<node {self.name}>"
