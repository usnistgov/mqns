from mqns.entity.cchannel import ClassicPacket, RecvClassicPacket
from mqns.entity.node import Application, Node
from mqns.network.network import QuantumNetwork
from mqns.network.protocol.classicforward import ClassicPacketForwardApp
from mqns.network.route import DijkstraRouteAlgorithm, RouteImpl
from mqns.network.topology import ClassicTopology, LinearTopology
from mqns.simulator import Simulator, func_to_event


class SendApp(Application):
    def __init__(self, dest: Node, route: RouteImpl, send_rate=1):
        super().__init__()
        self.dest = dest
        self.route = route
        self.send_rate = send_rate

    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        simulator.add_event(func_to_event(simulator.ts, self.send_packet, by=self))

    def send_packet(self):
        simulator = self.simulator

        packet = ClassicPacket(msg=f"Hello,world from {self.get_node()}", src=self.get_node(), dest=self.dest)

        route_result = self.route.query(self.get_node(), self.dest)
        if len(route_result) <= 0 or len(route_result[0]) <= 1:
            raise RuntimeError("not found next hop")
        next_hop = route_result[0][1]
        cchannel = self.get_node().get_cchannel(next_hop)

        # send the classic packet
        cchannel.send(packet=packet, next_hop=next_hop)

        # calculate the next sending time
        t = simulator.tc + 1 / self.send_rate

        # insert the next send event to the simulator
        event = func_to_event(t, self.send_packet, by=self)
        simulator.add_event(event)


# the receiving application
class RecvApp(Application):
    def __init__(self):
        super().__init__()
        self.add_handler(self.RecvClassicPacketHandler, RecvClassicPacket)

    def RecvClassicPacketHandler(self, event: RecvClassicPacket):
        packet = event.packet
        msg = packet.get()
        output = f"{self.get_node()} recv packet: {msg} from {packet.src}->{packet.dest}"
        print(output)
        assert output == "<qnode n10> recv packet: Hello,world from <qnode n1> from <qnode n1>-><qnode n10>"


def test_classic_forward():
    s = Simulator(0, 10, accuracy=10000000)

    topo = LinearTopology(nodes_number=10, qchannel_args={"delay": 0.1}, cchannel_args={"delay": 0.1})

    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)

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

    net.install(s)
    s.run()
