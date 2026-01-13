from typing import Any, override

from mqns.entity.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from mqns.entity.node import Node
from mqns.models.delay import NormalDelayModel, UniformDelayModel
from mqns.simulator import Event, Simulator, Time


class ClassicRecvNode(Node):
    def handle(self, event: Event) -> None:
        if isinstance(event, RecvClassicPacket):
            print(event.t, event.packet.src, event.packet.dest, event.packet.msg)


class ClassicSendNode(Node):
    def __init__(self, name: str, dest: Node):
        super().__init__(name=name)
        self.dest = dest

    def install(self, simulator: Simulator) -> None:
        super().install(simulator)

        t = 0
        while t < 10:
            time = simulator.time(sec=t)
            event = SendEvent(time, node=self, by=self)
            simulator.add_event(event)
            t += 0.25

    def send(self):
        print(self.simulator.tc, "send packet")
        link = self.cchannels[0]
        dest = self.dest
        packet = ClassicPacket(msg="ping", src=self, dest=dest)
        link.send(packet, dest)


class SendEvent(Event):
    def __init__(self, t: Time, node: ClassicSendNode, *, name: str | None = None, by: Any = None):
        super().__init__(t=t, name=name, by=by)
        self.node: ClassicSendNode = node

    @override
    def invoke(self) -> None:
        self.node.send()


def test_cchannel():
    n2 = ClassicRecvNode("n2")
    n1 = ClassicSendNode("n1", dest=n2)
    l1 = ClassicChannel(name="l1", bandwidth=10, delay=0.2, drop_rate=0.1, max_buffer_size=30)
    n1.add_cchannel(l1)
    n2.add_cchannel(l1)

    s = Simulator(0, 10, accuracy=1000, install_to=(n1, n2))
    s.run()


def test_cchannel_normal_delay():
    n2 = ClassicRecvNode("n2")
    n1 = ClassicSendNode("n1", dest=n2)
    l1 = ClassicChannel(
        name="l1", bandwidth=10, delay=NormalDelayModel(mean_delay=0.2, std=0.1), drop_rate=0.1, max_buffer_size=30
    )
    n1.add_cchannel(l1)
    n2.add_cchannel(l1)

    s = Simulator(0, 10, accuracy=1000, install_to=(n1, n2))
    s.run()


def test_cchannel_uniform_delay():
    n2 = ClassicRecvNode("n2")
    n1 = ClassicSendNode("n1", dest=n2)
    l1 = ClassicChannel(
        name="l1", bandwidth=10, delay=UniformDelayModel(min_delay=0.1, max_delay=0.3), drop_rate=0.1, max_buffer_size=30
    )
    n1.add_cchannel(l1)
    n2.add_cchannel(l1)

    s = Simulator(0, 10, accuracy=1000, install_to=(n1, n2))
    s.run()
