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
from typing import Any, Unpack, final, override

from mqns.entity.base_channel import BaseChannel, BaseChannelInitKwargs
from mqns.entity.node import Node
from mqns.simulator import Event, Time


class ClassicPacket:
    """ClassicPacket is the message that transfer on a ClassicChannel"""

    def __init__(self, msg: Any, *, src: Node, dest: Node):
        """Args:
        msg (Union[str, bytes, Any]): the message content.
            It can be a `str` or `bytes` type or can be dumpped to json.
        src (Node): the source of this message
        dest (Node): the destination of this message

        """
        self.is_json, self.msg = (False, msg) if isinstance(msg, (str, bytes)) else (True, json.dumps(msg))
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

    def get(self) -> Any:
        """Get the message from packet

        Return:
            (Union[str, bytes, Any])

        """
        return json.loads(self.msg) if self.is_json else self.msg

    def __len__(self) -> int:
        return len(self.msg)


class ClassicChannelInitKwargs(BaseChannelInitKwargs):
    pass


class ClassicChannel(BaseChannel[Node]):
    """ClassicChannel is the channel for classic message"""

    def __init__(self, name: str, **kwargs: Unpack[ClassicChannelInitKwargs]):
        super().__init__(name, **kwargs)

    @override
    def handle(self, event: Event):
        raise RuntimeError(f"unexpected event {event}")

    def send(self, packet: ClassicPacket, next_hop: Node):
        """Send a classic packet to the next_hop

        Args:
            packet (ClassicPacket): the packet
            next_hop (Node): the next hop Node
        Raises:
            NextHopNotConnectionException: the next_hop is not connected to this channel

        """
        drop, recv_time = self._send(
            packet_repr=f"packet {packet}",
            packet_len=len(packet),
            next_hop=next_hop,
        )

        if drop:
            return

        send_event = RecvClassicPacket(t=recv_time, by=self, cchannel=self, packet=packet, dest=next_hop)
        self.simulator.add_event(send_event)

    def __repr__(self) -> str:
        return "<cchannel " + self.name + ">"


@final
class RecvClassicPacket(Event):
    """The event for a Node to receive a classic packet"""

    def __init__(
        self, *, t: Time, name: str | None = None, by: Any = None, cchannel: ClassicChannel, packet: ClassicPacket, dest: Node
    ):
        super().__init__(t=t, name=name, by=by)
        self.cchannel = cchannel
        self.packet = packet
        self.dest = dest

    @override
    def invoke(self) -> None:
        self.dest.handle(self)
