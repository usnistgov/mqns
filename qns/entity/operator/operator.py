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

from typing import Callable

from qns.entity.entity import Entity
from qns.entity.node.qnode import QNode
from qns.entity.operator.event import OperateRequestEvent, OperateResponseEvent
from qns.models.delay import DelayInput, parseDelay
from qns.simulator.event import Event
from qns.simulator.simulator import Simulator


class QuantumOperator(Entity):
    """Quantum operator can perfrom quantum operation or measurements on qubits.
    It has two modes:
        Synchronous mode, users can use the `operate` function to operate qubits directly without delay
        Asynchronous mode, users will use events to operate quantum operations asynchronously
    """

    def __init__(self, name: str|None = None, *, node: QNode|None = None,
                 gate: Callable[..., None|int|list[int]], delay: DelayInput = 0):
        """Args:
        name (str): its name
        node (QNode): the quantum node that equips this memory
        gate: the quantum circuit where the input is the operating qubits and returns the measure result
        delay (Union[float,DelayModel]): the delay time in second for this operation or a ``DelayModel``

        """
        super().__init__(name=name)
        self.node = node
        self.gate = gate
        self.delay_model = parseDelay(delay)

    def install(self, simulator: Simulator) -> None:
        return super().install(simulator)

    def handle(self, event: Event) -> None:
        simulator = self.simulator

        if isinstance(event, OperateRequestEvent):
            assert self.node is not None

            qubits = event.qubits
            # operate qubits and get measure results
            result = self.operate(*qubits)

            t = simulator.tc + self.delay_model.calculate()
            response = OperateResponseEvent(node=self.node, result=result, request=event, t=t, by=self)
            simulator.add_event(response)

    def set_own(self, node: QNode):
        """Set the owner of this quantum operator
        """
        self.node = node

    def operate(self, *qubits) -> int|list[int]|None:
        """Operate on qubits and return the measure result

        Args:
            qubits: the operating qubits

        Returns:
            the measure result

        """
        return self.gate(*qubits)
