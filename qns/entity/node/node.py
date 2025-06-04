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

from qns.entity.entity import Entity
from qns.entity.node.app import Application
from qns.simulator import Event, Simulator

if TYPE_CHECKING:
    from qns.entity import ClassicChannel
    from qns.network import QuantumNetwork, SignalTypeEnum

ApplicationT = TypeVar("ApplicationT", bound=Application)

class Node(Entity):
    """Node is a generic node in the quantum network
    """

    def __init__(self, name: str, *, apps: list[Application]|None = None):
        """Args:
        name (str): the node's name
        apps (List[Application]): the installing applications.

        """
        super().__init__(name=name)
        self._network: "QuantumNetwork|None" = None
        self.cchannels: list["ClassicChannel"] = []
        self.croute_table = [] # XXX unused
        self.apps: list[Application] = [] if apps is None else apps

        # set default timing to ASYNC
        from qns.network.network import TimingModeEnum
        self.timing_mode: TimingModeEnum = TimingModeEnum.ASYNC

    def install(self, simulator: Simulator) -> None:
        """Called from Network.install()
        """
        super().install(simulator)
        # initiate sub-entities
        for cchannel in self.cchannels:
            from qns.entity import ClassicChannel
            assert isinstance(cchannel, ClassicChannel)
            cchannel.install(simulator)

        # initiate applications
        for app in self.apps:
            app.install(self, simulator)

    def handle(self, event: Event) -> None:
        """This function will handle an `Event`.
        This event will be passed to every applications in apps list in order.

        Args:
            event (Event): the event that happens on this QNode

        """
        for app in self.apps:
            skip = app.handle(self, event)
            if skip:
                break

    def add_apps(self, app: Application):
        """Insert an Application into the app list.
        Called from Topology.build() -> Topology._add_apps()

        Args:
            app (Application): the inserting application.

        """
        self.apps.append(app)

    def get_apps(self, app_type: type[ApplicationT]) -> list[ApplicationT]:
        """Get an Application that is `app_type`

        Args:
            app_type: the class of app_type

        """
        return [app for app in self.apps if isinstance(app, app_type)]

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
            network (qns.network.network.Network): the network object

        """
        self._network = network
        self.timing_mode = network.timing_mode

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

    def handle_sync_signal(self, signal_type: "SignalTypeEnum") -> None:
        for app in self.apps:
            app.handle_sync_signal(signal_type)

    def __repr__(self) -> str:
        if self.name is not None:
            return f"<node {self.name}>"
        return super().__repr__()
