import itertools
from typing import override

from mqns.entity.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from mqns.entity.node import Application, Node, QNode
from mqns.network.network import QuantumNetwork
from mqns.network.protocol.classicforward import ClassicPacketForwardApp
from mqns.network.route import DijkstraRouteAlgorithm, RouteAlgorithm
from mqns.simulator import Simulator, event_handler, func_to_event


class SendApp(Application[Node]):
    def __init__(self, dest: Node, route: RouteAlgorithm, send_rate=1):
        super().__init__()
        self.dest = dest
        self.route = route
        self.send_rate = send_rate

    @override
    def install(self, node):
        self._application_install(node, Node)
        self.simulator.sched(func_to_event(self.simulator.ts, self.send_packet))

    def send_packet(self):
        packet = ClassicPacket(msg=f"Hello,world from {self.node}", src=self.node, dest=self.dest)

        routes = self.route.query(self.node, self.dest)
        assert len(routes) > 0
        next_hop = routes[0].next_hop

        # send the classic packet
        self.node.send_cpacket(next_hop, packet)

        # calculate the next sending time
        t = self.simulator.tc + 1 / self.send_rate

        # insert the next send event to the simulator
        event = func_to_event(t, self.send_packet)
        self.simulator.sched(event)


# the receiving application
class RecvApp(Application[Node]):
    @event_handler
    def RecvClassicPacketHandler(self, event: RecvClassicPacket) -> None:
        packet = event.packet
        msg = packet.get()
        output = f"{self.node} recv packet: {msg} from {packet.src}->{packet.dest}"
        print(output)
        assert output == "<qnode n9> recv packet: Hello,world from <qnode n0> from <qnode n0>-><qnode n9>"


def test_classic_forward():
    net = QuantumNetwork()
    for i in range(10):
        net.add_node(QNode(f"n{i}"))
    for a, b in itertools.pairwise(net.nodes):
        ch = ClassicChannel(f"c_{a.name},{b.name}", delay=0.1)
        a.add_cchannel(ch)
        b.add_cchannel(ch)
        net.add_cchannel(ch)

    classic_route = DijkstraRouteAlgorithm(name="classic route")
    classic_route.build(list(net.nodes), net.cchannels)
    print(classic_route.route_table)

    for n in net.nodes:
        n.add_apps([ClassicPacketForwardApp(classic_route), RecvApp()])

    n0 = net.get_node("n0")
    n9 = net.get_node("n9")

    n0.add_apps(SendApp(n9, classic_route))

    s = Simulator(0, 10, accuracy=10000000, install_to=(net,))
    s.run()
