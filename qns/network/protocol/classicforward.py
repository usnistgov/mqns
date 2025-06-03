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

from qns.entity.cchannel import RecvClassicPacket
from qns.entity.node import Application, Node
from qns.network.route import RouteImpl


class ClassicPacketForwardApp(Application):
    """This application will generate routing table for classic networks
    and allow nodes to forward classic packets to the destination.
    """

    def __init__(self, route: RouteImpl):
        """Args:
        route (RouteImpl): a route implement

        """
        super().__init__()
        self.route = route
        self.add_handler(self.handleClassicPacket, [RecvClassicPacket], [])

    def handleClassicPacket(self, node: Node, event: RecvClassicPacket):
        packet = event.packet
        self_node = self.get_node()

        dst = packet.dest
        if dst == self_node:
            # The destination is this node, return to let later application to handle this packet
            return False

        # If destination is not this node, forward this packet
        route_result = self.route.query(self.get_node(), dst)
        if len(route_result) <= 0 or len(route_result[0]) <= 1:
            # no routing result or error format, drop this packet
            return True
        next_hop = route_result[0][1]
        cchannel = self_node.get_cchannel(next_hop)
        if cchannel is None:
            # not found the classic channel, drop the packet
            return True
        cchannel.send(packet=packet, next_hop=next_hop)
        return True
