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
from typing import TYPE_CHECKING, Any, cast

from mqns.simulator import EventDispatcherMixin

if TYPE_CHECKING:
    from mqns.entity.node.node import Node


class Application[N: "Node"](EventDispatcherMixin, ABC):
    """
    Application deployed on a node.

    ``N`` type parameter indicates which ``Node`` subclass is required for installing this application.
    """

    def install(self, node: "Node"):
        """
        Install this application onto the node.

        Base class implementation does not verify ``node`` matches ``N`` type parameter.
        If ``N`` type parameter is a subclass such as ``QNode``, subclass should override this method to
        invoke ``self._application_install()`` with an appropriate ``node_type``.
        """
        from mqns.entity.node.node import Node  # noqa: PLC0415

        self._application_install(node, cast(Any, Node))

    def _application_install(self, node: "Node", node_type: type[N]) -> None:
        """
        Part of ``install`` method logic.
        """
        self.simulator = node.simulator
        """Global simulator instance."""
        assert isinstance(node, node_type)
        self.node: N = node
        """Node that owns this application."""

    def __repr__(self: Any) -> str:
        """
        Represent self as "<Type node-name>".

        This may be reused on any type that has an optional ``.node`` member.
        """
        try:
            return f"<{type(self).__name__} {self.node.name}>"
        except AttributeError:  # self.node does not exist before .install()
            return object.__repr__(self)
