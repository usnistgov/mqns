from typing import Any, cast

from typing_extensions import override

from mqns.entity.node import Application, Node
from mqns.network.protocol import NodeProcessDelayApp
from mqns.simulator.event import Event
from mqns.simulator.simulator import Simulator
from mqns.simulator.ts import Time


class ProcessEvent(Event):
    def __init__(self, t: Time, dest: Node, name: str | None = None, by: Any = None):
        super().__init__(t, name=name, by=by)
        self.dest = dest

    @override
    def invoke(self) -> None:
        self.dest.handle(self)


class ProcessApp(Application):
    def __init__(self):
        super().__init__()
        self.add_handler(self.EventHandler)

    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)

        for i in range(0, 10):
            t = simulator.time(sec=i)
            event = ProcessEvent(t=t, dest=self.get_node(), by=self)
            self.simulator.add_event(event)

    def EventHandler(self, event: Event) -> bool | None:
        expected_recv_time = [i + 0.5 for i in range(0, 10)]
        print(f"recv event at {event.t}")
        assert cast(Time, event.t).sec in expected_recv_time


def test_process_delay():
    n1 = Node("n1")
    n1.add_apps(NodeProcessDelayApp(delay=0.5, delay_event_list=(ProcessEvent,)))
    n1.add_apps(ProcessApp())

    s = Simulator(0, 10)
    n1.install(s)

    s.run()
