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
from qns.entity.node.node import Node


class Controller(Node):
    """
    Controller computes routing and swapping instructions for quantum routers 
    """
    def __init__(self, name: str = None, apps: List[Application] = None):
        """
        Args:
            name (str): the node's name
            apps (List[Application]): the installing applications.
        """
        super().__init__(name=name, apps=apps)

    def install(self, simulator: Simulator) -> None:
        super().install(simulator)
        # do other things specific to the controller

    def __repr__(self) -> str:
        if self.name is not None:
            return f"<controller {self.name}>"
        return super().__repr__()
