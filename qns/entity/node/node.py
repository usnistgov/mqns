#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Amar Abane
#    National Institute of Standards and Technology, NIST.
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

from typing import List, Union
from qns.simulator import Simulator
from qns.simulator import Event
from qns.entity import Entity
from qns.entity.node.app import Application


class Node(Entity):
    """
    Node is a generic node in the quantum network
    """
    def __init__(self, name: str = None, apps: List[Application] = None):
        """
        Args:
            name (str): the node's name
            apps (List[Application]): the installing applications.
        """
        super().__init__(name=name)
        self.network = None
        self.cchannels = []
        self.croute_table = []
        if apps is None:
            self.apps: List[Application] = []
        else:
            self.apps: List[Application] = apps
        
        # set default timing to ASYNC
        from qns.network.network import TimingModeEnum
        self.timing_mode = TimingModeEnum.ASYNC

    def install(self, simulator: Simulator) -> None: 
        """
        Called from Network.install()
        """
        super().install(simulator)
        # initiate sub-entities
        for cchannel in self.cchannels:
            from qns.entity import ClassicChannel
            assert (isinstance(cchannel, ClassicChannel))
            cchannel.install(simulator)

        # initiate applications
        for app in self.apps:
            app.install(self, simulator)

    def handle(self, event: Event) -> None:
        """
        This function will handle an `Event`.
        This event will be passed to every applications in apps list in order.

        Args:
            event (Event): the event that happens on this QNode
        """
        for app in self.apps:
            skip = app.handle(self, event)
            if skip:
                break

    def add_apps(self, app: Application):
        """
        Insert an Application into the app list. 
        Called from Topology.build() -> Topology._add_apps()

        Args:
            app (Application): the inserting application.
        """
        self.apps.append(app)

    def get_apps(self, app_type):
        """
        Get an Application that is `app_type`

        Args:
            app_type: the class of app_type
        """
        return [app for app in self.apps if isinstance(app, app_type)]

    def add_cchannel(self, cchannel):
        """
        Add a classic channel in this Node

        Args:
            cchannel (ClassicChannel): the classic channel
        """
        cchannel.node_list.append(self)
        self.cchannels.append(cchannel)

    def get_cchannel(self, dst: "Node"):
        """
        Get the classic channel that connects to the `dst`

        Args:
            dst (Node): the destination
        """
        for cchannel in self.cchannels:
            if dst in cchannel.node_list and self in cchannel.node_list:
                return cchannel
        return None

    def add_network(self, network):
        """
        add a network object to this node. 
        Called from Network.__init__()

        Args:
            network (qns.network.network.Network): the network object
        """
        self.network = network
        self.timing_mode = network.timing_mode
        
    def handle_sync_signal(self, signal_type) -> None:
        for app in self.apps:
            app.handle_sync_signal(signal_type)

    def __repr__(self) -> str:
        if self.name is not None:
            return f"<node {self.name}>"
        return super().__repr__()
