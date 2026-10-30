from collections.abc import Callable, Sequence
from typing import Any, Final, TypedDict, Unpack

import pytest

from mqns.entity.cchannel import (
    ClassicChannel,
    ClassicChannelInitKwargs,
    ClassicCommandDispatcherMixin,
    ClassicCommandModule,
    ClassicPacket,
    RecvClassicPacket,
    classic_cmd_handler,
)
from mqns.entity.node import Application, Node
from mqns.models.delay import UniformDelayModel
from mqns.simulator import Simulator, event_handler, func_to_event


class SendRecvApp(Application[Node]):
    def __init__(self):
        super().__init__()

        self.n_tx = 0
        self.rx: list[RecvClassicPacket] = []

    @event_handler
    def handle_recv(self, event: RecvClassicPacket) -> None:
        self.rx.append(event)

    def send(self, dest: Node, msgs: list[Any]):
        for msg in msgs:
            self.n_tx += 1
            self.node.send_cpacket(dest, ClassicPacket(msg, src=self.node, dest=dest))

    def sched_sends(self, n: int, dest: Node, since: float, interval: float, msgs: Callable[[int], list[Any]] = lambda _: [""]):
        t = self.simulator.time(sec=since)
        for i in range(n):
            event = func_to_event(t, self.send, dest, list(msgs(i)))
            self.simulator.sched(event)
            t += interval


def make_nodes[T: Application[Node]](n=2, app_type: type[T] = SendRecvApp, **kwargs: Unpack[ClassicChannelInitKwargs]):
    cc = ClassicChannel("cc", **kwargs)
    nodes: list[Node] = []
    apps: list[T] = []
    for i in range(n):
        node = Node(f"n{i}")
        node.add_cchannel(cc)
        app = app_type()
        node.add_apps(app)
        nodes.append(node)
        apps.append(app)
    simulator = Simulator(0, 10, accuracy=1000, install_to=nodes)
    return simulator, *apps


def test_delay_const():
    simulator, a0, a1 = make_nodes(delay=0.2)
    a0.sched_sends(1000, a1.node, 1.000, 0.001, lambda i: [{"A": i}])
    simulator.run()

    assert a0.n_tx == 1000
    assert len(a1.rx) == 1000
    for i, event in enumerate(a1.rx):
        assert event.t == simulator.time(sec=1.200 + i / 1000)
        assert event.packet.get() == {"A": i}


def test_delay_uniform():
    simulator, a0, a1 = make_nodes(delay=UniformDelayModel(min=0.1, max=0.5))
    a0.sched_sends(1000, a1.node, 1.000, 0.000)
    simulator.run()

    assert a0.n_tx == 1000
    assert len(a1.rx) == 1000

    min_t = min(event.t for event in a1.rx)
    max_t = max(event.t for event in a1.rx)
    assert 1.100 <= min_t.sec < 1.150
    assert 1.450 < max_t.sec <= 1.500


def test_drop():
    simulator, a0, a1 = make_nodes(drop_rate=0.3)
    a0.sched_sends(1000, a1.node, 1.000, 0.001)
    simulator.run()

    assert a0.n_tx == 1000
    assert 500 < len(a1.rx) < 900


def test_bandwidth():
    # 4 octets per packet, 1 packets allowed per 0.010 seconds, 2 packets can be buffered
    simulator, a0, a1 = make_nodes(bandwidth=400, max_buffer_size=8)
    # sending 100 packets in 0.1 seconds, 10 are sent on time, 2 are buffered
    a0.sched_sends(100, a1.node, 1.000, 0.001, lambda i: [f"{i:04}"])
    simulator.run()

    assert a0.n_tx == 100
    assert [(int(event.t.sec * 1000), event.packet.get()) for event in a1.rx] == [
        (1000, "0000"),  # sent on time
        (1010, "0001"),  # queued
        (1020, "0002"),  # queued
        (1030, "0010"),  # admitted into queue when "0001" is sent
        (1040, "0020"),  # admitted into queue when "0002" is sent
        (1050, "0030"),
        (1060, "0040"),
        (1070, "0050"),
        (1080, "0060"),
        (1090, "0070"),
        (1100, "0080"),
        (1110, "0090"),
    ]
    assert len(a1.rx) == 12


class SimpleCommand(TypedDict):
    cmd: str


class CommandAppA(ClassicCommandDispatcherMixin, Application[Node]):
    def __init__(self):
        super().__init__()
        self.records = set[str]()

    def save(self, pkt: ClassicPacket, msg: SimpleCommand) -> None:
        _ = pkt
        self.records.add(msg["cmd"])

    @classic_cmd_handler("a")
    def handle_a(self, pkt: ClassicPacket, msg: SimpleCommand) -> None:
        self.save(pkt, msg)


class CommandAppB(CommandAppA):
    @classic_cmd_handler("b")
    def handle_b(self, pkt: ClassicPacket, msg: SimpleCommand) -> None:
        self.save(pkt, msg)


class CommandAppC(CommandAppA):
    @classic_cmd_handler("c")
    def handle_c(self, pkt: ClassicPacket, msg: SimpleCommand) -> None:
        self.save(pkt, msg)


class CommandAppD(CommandAppB, CommandAppC):
    @classic_cmd_handler("d")
    def handle_d(self, pkt: ClassicPacket, msg: SimpleCommand) -> None:
        self.save(pkt, msg)


class CommandModuleE(ClassicCommandModule):
    def __init__(self, owner: CommandAppA):
        self.owner = owner

    @classic_cmd_handler("e")
    def handle_e(self, pkt: ClassicPacket, msg: SimpleCommand) -> None:
        self.owner.save(pkt, msg)


class CommandAppE(CommandAppA):
    module_e: CommandModuleE

    def __init__(self):
        super().__init__()
        self.module_e = CommandModuleE(self)


class CommandModuleF(CommandModuleE):
    @classic_cmd_handler("f")
    def handle_f(self, pkt: ClassicPacket, msg: SimpleCommand) -> None:
        self.owner.save(pkt, msg)


class CommandAppF(CommandAppA):
    module_f: Final[CommandModuleF]
    not_module_int: int
    not_module_final_int: Final[int]
    not_module_sequence: Sequence[str]
    not_module_final_sequence: Final[Sequence[str]]

    def __init__(self):
        super().__init__()
        self.module_f = CommandModuleF(self)
        self.not_module_int = 0
        self.not_module_final_int = 0
        self.not_module_sequence = []
        self.not_module_final_sequence = []


@pytest.mark.parametrize(
    ("app_type", "expected"),
    [
        (CommandAppB, {"a", "b"}),
        (CommandAppC, {"a", "c"}),
        (CommandAppD, {"a", "b", "c", "d"}),
        (CommandAppE, {"a", "e"}),
        (CommandAppF, {"a", "e", "f"}),
    ],
)
def test_dispatch(app_type: type[CommandAppA], expected: dict[str, int]):
    simulator, a0, a1 = make_nodes(app_type=app_type)

    def send(cmd: str):
        msg = SimpleCommand(cmd=cmd)
        pkt = ClassicPacket(msg, src=a0.node, dest=a1.node)
        a0.node.send_cpacket(a1.node, pkt)

    send("a")
    send("b")
    send("c")
    send("d")
    send("e")
    send("f")

    simulator.run()
    assert a1.records == expected
