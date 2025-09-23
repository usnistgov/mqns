#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
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

from mqns.entity.node.app import Application
from mqns.entity.node.node import Node
from mqns.simulator import Simulator


class Controller(Node):
    """Controller computes routing and swapping instructions for quantum routers"""

    def __init__(self, name: str, *, apps: list[Application] | None = None):
        """Args:
        name (str): the node's name
        apps (List[Application]): the installing applications.

        """
        super().__init__(name=name, apps=apps)

    def install(self, simulator: Simulator) -> None:
        super().install(simulator)
        # do other things specific to the controller

    def __repr__(self) -> str:
        return f"<controller {self.name}>"
