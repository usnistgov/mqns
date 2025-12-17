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

import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from mqns.simulator import Event, Simulator

if TYPE_CHECKING:
    from mqns.entity.node.node import Node, NodeT

EventT = TypeVar("EventT", bound=Event)
"""Represents an event type."""


class Application:
    """
    Application deployed on a node.
    """

    def __init__(self):
        self._simulator: Simulator | None = None
        self._node: "Node|None" = None
        self._dispatch_table = defaultdict[type[Event], list[tuple[set[Any] | None, Callable[[Event], bool | None]]]](
            lambda: []
        )

    def install(self, node: "Node", simulator: Simulator):
        """Install initial events for this Node. Called from Node.install()

        Args:
            node (Node): the node that will run this application
            simulator (Simulator): the simulator

        """
        self._simulator = simulator
        self._node = node

    def handle(self, event: Event) -> bool | None:
        """
        Dispatch an event in the application.

        Args:
            event: the event

        Return:
            skip (bool, None): if True, further applications will not handle this event
        """
        return self._dispatch(event)

    def _dispatch(self, event: Event) -> bool:
        for eb, handler in self._dispatch_table.get(type(event), []):
            if eb is None or event.by in eb:
                skip = handler(event)
                if skip is True:
                    return skip
        return False

    def add_handler(
        self,
        handler: Callable[[EventT], bool | None],
        event_type: type[EventT] | Iterable[type[EventT]],
        event_by: Iterable[Any] | None = None,
    ):
        """
        Add an event handler function.

        Args:
            handler: Event handler function.
            event_type: Event type(s). Each class must be marked `@final`.
            event_by: filter by event source entity (`event.by`), defaults to any source.
        """
        ets = [event_type] if isinstance(event_type, type) else event_type
        eb = None if event_by is None else set(event_by)
        eh = cast(Any, handler)
        for et in cast(Iterable[type[EventT]], ets):
            # __final__ marker is available since Python 3.11
            assert sys.version_info[:2] < (3, 11) or getattr(et, "__final__", False) is True, (
                f"event type {et} must be marked @final"
            )
            self._dispatch_table[et].append((eb, eh))

    @overload
    def get_node(self) -> "Node":
        pass

    @overload
    def get_node(self, *, node_type: type["NodeT"]) -> "NodeT":
        pass

    def get_node(self, *, node_type: type["NodeT"] | None = None):
        """
        Retrieve the owner node, optionally asserts its type.

        Raises:
            IndexError - application is not installed.
            TypeError - owner node has wrong type.
        """
        if self._node is None:
            raise IndexError("application is not in a node")
        if node_type is not None and not isinstance(self._node, node_type):
            raise TypeError(f"application owner node is not of type {node_type}")
        return cast("NodeT", self._node)

    @property
    def simulator(self) -> Simulator:
        """
        Retrieve the simulator.

        Raises:
            IndexError - application is not installed.
        """
        if self._simulator is None:
            raise IndexError("application is not in a simulator")
        return self._simulator


ApplicationT = TypeVar("ApplicationT", bound=Application)
"""Represents an application type."""
