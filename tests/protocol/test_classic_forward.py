from typing import override

from mqns.entity.cchannel import ClassicPacket, RecvClassicPacket
from mqns.entity.node import Application, Node
from mqns.network.network import QuantumNetwork
from mqns.network.protocol.classicforward import ClassicPacketForwardApp
from mqns.network.route import DijkstraRouteAlgorithm, RouteAlgorithm
from mqns.network.topology import ClassicTopology, LinearTopology
from mqns.simulator import Simulator, func_to_event


class SendApp(Application[Node]):
    def __init__(self, dest: Node, route: RouteAlgorithm, send_rate=1):
        super().__init__()
        self.dest = dest
        self.route = route
        self.send_rate = send_rate

    @override
    def install(self, node):
        self._application_install(node, Node)
        self.simulator.add_event(func_to_event(self.simulator.ts, self.send_packet, by=self))

    def send_packet(self):
        packet = ClassicPacket(msg=f"Hello,world from {self.node}", src=self.node, dest=self.dest)

        route_result = self.route.query(self.node, self.dest)
        if len(route_result) <= 0 or len(route_result[0]) <= 1:
            raise RuntimeError("not found next hop")
        next_hop = route_result[0][1]
        cchannel = self.node.get_cchannel(next_hop)

        # send the classic packet
        cchannel.send(packet=packet, next_hop=next_hop)

        # calculate the next sending time
        t = self.simulator.tc + 1 / self.send_rate

        # insert the next send event to the simulator
        event = func_to_event(t, self.send_packet, by=self)
        self.simulator.add_event(event)


# the receiving application
class RecvApp(Application[Node]):
    def __init__(self):
        super().__init__()
        self.add_handler(self.RecvClassicPacketHandler, RecvClassicPacket)

    def RecvClassicPacketHandler(self, event: RecvClassicPacket):
        packet = event.packet
        msg = packet.get()
        output = f"{self.node} recv packet: {msg} from {packet.src}->{packet.dest}"
        print(output)
        assert output == "<qnode n10> recv packet: Hello,world from <qnode n1> from <qnode n1>-><qnode n10>"


def test_classic_forward():
    topo = LinearTopology(nodes_number=10, qchannel_args={"delay": 0.1}, cchannel_args={"delay": 0.1})

    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow)

    # build quantum routing table
    net.build_route()

    classic_route = DijkstraRouteAlgorithm(name="classic route")

    # build classic routing table
    classic_route.build(net.nodes, net.cchannels)
    print(classic_route.route_table)

    for n in net.nodes:
        n.add_apps(ClassicPacketForwardApp(classic_route))
        n.add_apps(RecvApp())

    n1 = net.get_node("n1")
    n10 = net.get_node("n10")

    n1.add_apps(SendApp(n10, classic_route))

    s = Simulator(0, 10, accuracy=10000000, install_to=(net,))
    s.run()
