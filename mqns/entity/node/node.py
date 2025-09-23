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

from typing import TYPE_CHECKING, TypeVar

from typing_extensions import override

from mqns.entity.entity import Entity
from mqns.entity.node.app import Application
from mqns.simulator import Event, Simulator

if TYPE_CHECKING:
    from mqns.entity.cchannel import ClassicChannel
    from mqns.network.network import QuantumNetwork, TimingMode

ApplicationT = TypeVar("ApplicationT", bound=Application)


class Node(Entity):
    """Node is a generic node in the quantum network"""

    def __init__(self, name: str, *, apps: list[Application] | None = None):
        """Args:
        name (str): the node's name
        apps (List[Application]): the installing applications.

        """
        super().__init__(name=name)
        self._network: "QuantumNetwork|None" = None
        self.cchannels: list["ClassicChannel"] = []
        self.apps: list[Application] = [] if apps is None else apps

    def install(self, simulator: Simulator) -> None:
        """Called from Network.install()"""
        super().install(simulator)
        # initiate sub-entities
        for cchannel in self.cchannels:
            from mqns.entity import ClassicChannel  # noqa: PLC0415

            assert isinstance(cchannel, ClassicChannel)
            cchannel.install(simulator)

        # initiate applications
        for app in self.apps:
            app.install(self, simulator)

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
            app_type: the class of app_type

        """
        return [app for app in self.apps if isinstance(app, app_type)]

    def get_app(self, app_type: type[ApplicationT]) -> ApplicationT:
        """
        Retrieve an application of given type.
        There must be exactly one instance of this application.

        Args:
            app_type: the class of app_type

        Raises:
            IndexError
        """
        apps = self.get_apps(app_type)
        if len(apps) != 1:
            raise IndexError(f"node {repr(self)} has {len(apps)} instances of {app_type}")
        return apps[0]

    def add_cchannel(self, cchannel: "ClassicChannel"):
        """Add a classic channel in this Node

        Args:
            cchannel (ClassicChannel): the classic channel

        This function is available prior to calling .install().
        """
        assert self._simulator is None
        cchannel.node_list.append(self)
        self.cchannels.append(cchannel)

    def get_cchannel(self, dst: "Node") -> "ClassicChannel":
        """Get the classic channel that connects to the `dst`

        Args:
            dst (Node): the destination

        Raises:
            IndexError - channel does not exist
        """
        for cchannel in self.cchannels:
            if dst in cchannel.node_list and self in cchannel.node_list:
                return cchannel
        raise IndexError(f"cchannel from {repr(self)} to {repr(dst)} does not exist")

    def add_network(self, network: "QuantumNetwork"):
        """Add a network object to this node.
        Called from Network.__init__()

        Args:
            network (mqns.network.network.Network): the network object

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
