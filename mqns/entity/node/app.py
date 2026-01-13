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

from abc import ABC
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from mqns.simulator import Event, EventT

if TYPE_CHECKING:
    from mqns.entity.node.node import Node


NodeT = TypeVar("NodeT", bound="Node")
"""Type argument for Node or its subclass."""


class Application(ABC, Generic[NodeT]):
    """
    Application deployed on a node.

    ``NodeT`` parameter indicates which ``Node`` subclass is required for installing this application.
    """

    def __init__(self):
        self._dispatch_table = defaultdict[type[Event], list[tuple[set[Any] | None, Callable[[Event], bool | None]]]](
            lambda: []
        )

    def install(self, node: "Node"):
        """
        Install this application onto the node.

        Base class implementation does not verify ``node`` matches ``NodeT`` type.
        If ``NodeT`` is a subclass such as ``QNode``, subclass should override this method to
        invoke ``self._application_install()`` with an appropriate ``node_type``.
        """
        from mqns.entity.node.node import Node  # noqa: PLC0415

        self._application_install(node, cast(Any, Node))

    def _application_install(self, node: "Node", node_type: type[NodeT]) -> None:
        """
        Part of ``install`` method logic.
        """
        self.simulator = node.simulator
        """Global simulator instance."""
        assert isinstance(node, node_type)
        self.node: NodeT = node
        """Node that owns this application."""

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
            assert getattr(et, "__final__", False) is True, f"event type {et} must be marked @final"
            self._dispatch_table[et].append((eb, eh))


ApplicationT = TypeVar("ApplicationT", bound=Application)
"""Represents an application type."""
