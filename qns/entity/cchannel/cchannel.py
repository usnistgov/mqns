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

import json
from typing import Any, TypedDict

from qns.entity.entity import Entity
from qns.entity.node import Node
from qns.models.delay import DelayInput, parseDelay
from qns.simulator import Event, Simulator, Time
from qns.utils import get_rand, log

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


class ClassicPacket:
    """ClassicPacket is the message that transfer on a ClassicChannel"""

    def __init__(self, msg: Any, src: Node | None = None, dest: Node | None = None):
        """Args:
        msg (Union[str, bytes, Any]): the message content.
            It can be a `str` or `bytes` type or can be dumpped to json.
        src (Node): the source of this message
        dest (Node): the destination of this message

        """
        self.is_json: bool = False
        # if not isinstance(msg, (str, bytes)):
        #    self.msg = json.dumps(msg)
        #    self.is_json = True
        # else:
        self.msg = msg
        self.src = src
        self.dest = dest

    def encode(self) -> bytes:
        """Encode the self.msg if it is a `str`

        Return:
            (bytes) a `bytes` object

        """
        if isinstance(self.msg, str):
            return self.msg.encode(encoding="utf-8")
        assert isinstance(self.msg, bytes)
        return self.msg

    def get(self):
        """Get the message from packet

        Return:
            (Union[str, bytes, Any])

        """
        if self.is_json:
            return json.loads(self.msg)
        return self.msg

    def __len__(self) -> int:
        return len(self.msg)


class ClassicChannelInitKwargs(TypedDict, total=False):
    bandwidth: int
    delay: DelayInput
    drop_rate: float
    max_buffer_size: int
    length: float


class ClassicChannel(Entity):
    """ClassicChannel is the channel for classic message"""

    def __init__(self, name: str, node_list: list[Node] = [], **kwargs: Unpack[ClassicChannelInitKwargs]):
        """Args:
        name (str): the name of this channel
        node_list (List[Node]): a list of QNodes that it connects to
        bandwidth (int): the byte per second on this channel. 0 represents unlimited
        delay (Union[float, DelayModel]): the time delay for transmitting a packet. It is a float number or a ``DelayModel``
        length (float): the length of this channel
        drop_rate (float): the drop rate
        max_buffer_size (int): the max buffer size.
            If it is full, the next coming packet will be dropped. 0 represents unlimited.

        """
        super().__init__(name=name)
        self.node_list = node_list.copy()
        self.bandwidth = kwargs.get("bandwidth", 0)
        self.delay_model = parseDelay(kwargs.get("delay", 0))
        self.drop_rate = kwargs.get("drop_rate", 0.0)
        assert 0.0 <= self.drop_rate <= 1.0
        self.max_buffer_size = kwargs.get("max_buffer_size", 0)
        self.length = kwargs.get("length", 0.0)
        self._next_send_time: Time

    def install(self, simulator: Simulator) -> None:
        """``install`` is called before ``simulator`` runs to initialize or set initial events

        Args:
            simulator (qns.simulator.simulator.Simulator): the simulator

        """
        super().install(simulator)
        self._next_send_time = simulator.ts

    def send(self, packet: ClassicPacket, next_hop: Node, delay: float = 0):
        """Send a classic packet to the next_hop

        Args:
            packet (ClassicPacket): the packet
            next_hop (Node): the next hop Node
        Raises:
            qns.entity.cchannel.cchannel.NextHopNotConnectionException:
                the next_hop is not connected to this channel

        """
        simulator = self.simulator

        if next_hop not in self.node_list:
            raise NextHopNotConnectionException

        if self.bandwidth != 0:
            if self._next_send_time <= simulator.current_time:
                send_time = simulator.current_time
            else:
                send_time = self._next_send_time

            if self.max_buffer_size != 0 and send_time > simulator.current_time + self.max_buffer_size / self.bandwidth:
                # buffer is overflow
                log.debug(f"cchannel {self}: drop packet {packet} due to overflow")
                return

            self._next_send_time = send_time + len(packet) / self.bandwidth
        else:
            send_time = simulator.current_time

        # random drop
        if get_rand() < self.drop_rate:
            log.debug(f"cchannel {self}: drop packet {packet} due to drop rate")
            return
        #  add delay
        recv_time = send_time + (self.delay_model.calculate() + delay)

        send_event = RecvClassicPacket(t=recv_time, name=None, by=self, cchannel=self, packet=packet, dest=next_hop)
        simulator.add_event(send_event)

    def __repr__(self) -> str:
        return "<cchannel " + self.name + ">"


class NextHopNotConnectionException(Exception):
    pass


class RecvClassicPacket(Event):
    """The event for a Node to receive a classic packet"""

    def __init__(
        self, *, t: Time, name: str | None = None, by: Any = None, cchannel: ClassicChannel, packet: ClassicPacket, dest: Node
    ):
        super().__init__(t=t, name=name, by=by)
        self.cchannel = cchannel
        self.packet = packet
        self.dest = dest

    def invoke(self) -> None:
        self.dest.handle(self)
