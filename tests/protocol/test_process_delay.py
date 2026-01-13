from typing import Any, cast, final, override

from mqns.entity.node import Application, Node
from mqns.network.protocol import NodeProcessDelayApp
from mqns.simulator import Event, Simulator, Time


@final
class ProcessEvent(Event):
    def __init__(self, t: Time, dest: Node, name: str | None = None, by: Any = None):
        super().__init__(t, name=name, by=by)
        self.dest = dest

    @override
    def invoke(self) -> None:
        self.dest.handle(self)


class ProcessApp(Application[Node]):
    def __init__(self):
        super().__init__()
        self.add_handler(self.EventHandler, ProcessEvent)

    @override
    def install(self, node):
        self._application_install(node, Node)

        for i in range(0, 10):
            t = self.simulator.time(sec=i)
            event = ProcessEvent(t=t, dest=self.node, by=self)
            self.simulator.add_event(event)

    def EventHandler(self, event: Event) -> bool | None:
        expected_recv_time = [i + 0.5 for i in range(0, 10)]
        print(f"recv event at {event.t}")
        assert cast(Time, event.t).sec in expected_recv_time


def test_process_delay():
    n1 = Node("n1")
    n1.add_apps(NodeProcessDelayApp(delay=0.5, delay_event_list=(ProcessEvent,)))
    n1.add_apps(ProcessApp())

    s = Simulator(0, 10, install_to=(n1,))
    s.run()
