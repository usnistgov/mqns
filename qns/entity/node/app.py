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

from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

from qns.simulator import Event, Simulator

if TYPE_CHECKING:
    from qns.entity.node.node import Node
    from qns.entity.node.qnode import QNode
    from qns.network import SignalTypeEnum

EventT = TypeVar("EventT")
Handler = Callable[["Node", EventT], Any] | Callable[["QNode", EventT], bool|None]

class Application:
    """Application can be deployed on the quantum nodes.
    """

    def __init__(self):
        self._simulator: Simulator|None = None
        self._node: "Node|None" = None
        self._dispatch_dict: list[tuple[list[type[Event]],list[Any],Handler]] = []

    def install(self, node: "Node", simulator: Simulator):
        """Install initial events for this Node. Called from Node.install()

        Args:
            node (Node): the node that will handle this event
            simulator (Simulator): the simulator

        """
        self._simulator = simulator
        self._node = node

    def handle(self, node: "Node", event: Event) -> bool|None:
        """Process the event on the node.

        Args:
            node (Node): the node that will handle this event
            event (Event): the event

        Return:
            skip (bool, None): if skip is True, further applications will not handle this event

        """
        return self._dispatch(node, event)

    def _dispatch(self, node: "Node", event: Event) -> bool|None:
        for eventTypeList, byList, handler in self._dispatch_dict:
            flag_et = False
            flag_by = False
            if len(eventTypeList) > 0:
                for et in eventTypeList:
                    if isinstance(event, et):
                        flag_et = True
            else:
                flag_et = True

            if len(byList) == 0 or event.by in byList:
                flag_by = True

            if flag_et and flag_by:
                skip = handler(cast(Any, node), event)
                if skip is True:
                    return skip
        return False

    def add_handler(self, handler: Handler, EventTypeList: list[type[Event]] = [], ByList: list[Any] = []):
        """Add a handler function to the dispather.

        Args:
            handler: The handler to process the event.
                It is an object method whose function signature is the same to ``handle`` function.
            EventTypeList: a list of Event Class Type. An empty list meaning to match all events.
            ByList: a list of Entities, QNodes or Applications, that generates this event.
                An empty list meaning to match all entities.

        """
        self._dispatch_dict.append((EventTypeList, ByList, handler))

    def get_node(self) -> "Node":
        """Get the node that runs this application

        Returns:
            the quantum node

        """
        if self._node is None:
            raise IndexError("application is not in a node")
        return self._node

    def get_simulator(self) -> Simulator:
        """Get the simulator

        Returns:
            the simulator

        """
        if self._simulator is None:
            raise IndexError("application is not in a simulator")
        return self._simulator

    def handle_sync_signal(self, signal_type: "SignalTypeEnum"):
        pass
