from mqns.entity.node import Application, Controller
from mqns.network.network import QuantumNetwork, TimingModeSync, TimingPhase, TimingPhaseEvent
from mqns.network.topology import BasicTopology
from mqns.simulator import Simulator


class SyncCheckApp(Application):
    def __init__(self):
        super().__init__()
        self.changes = 0
        self.add_handler(self.handle_sync_signal, TimingPhaseEvent)

    def handle_sync_signal(self, event: TimingPhaseEvent):
        t = self.simulator.tc.sec
        assert (t % 5 == 0 and event.phase == TimingPhase.EXTERNAL) or (t % 5 == 4 and event.phase == TimingPhase.INTERNAL)
        self.changes += 1


def test_timing_mode_sync():
    topo = BasicTopology(2, nodes_apps=[SyncCheckApp()])
    topo.controller = Controller("ctrl", apps=[SyncCheckApp()])
    net = QuantumNetwork(topo=topo, timing=TimingModeSync(t_ext=4, t_int=1))

    simulator = Simulator(start_second=0.0, end_second=29.9)
    net.install(simulator)
    simulator.run()

    for node in net.nodes + [net.get_controller()]:
        app = node.get_app(SyncCheckApp)
        assert app.changes == 12
