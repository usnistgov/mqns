from mqns.entity.node import Application, Controller, Node
from mqns.network.network import QuantumNetwork, TimingModeSync, TimingPhase, TimingPhaseEvent
from mqns.network.topology import BasicTopology
from mqns.simulator import Simulator


class SyncCheckApp(Application[Node]):
    def __init__(self):
        super().__init__()
        self.enters = 0
        self.exits = 0
        self.add_handler(self.handle_sync_signal, TimingPhaseEvent)

    def handle_sync_signal(self, event: TimingPhaseEvent):
        t = self.simulator.tc.sec
        if event.enter:
            assert (t % 5 == 0 and event.phase is TimingPhase.EXTERNAL) or (t % 5 == 4 and event.phase is TimingPhase.INTERNAL)
            self.enters += 1
        else:
            assert (t % 5 == 0 and event.phase is TimingPhase.INTERNAL) or (t % 5 == 4 and event.phase is TimingPhase.EXTERNAL)
            self.exits += 1


def test_timing_mode_sync():
    topo = BasicTopology(2, nodes_apps=[SyncCheckApp()])
    topo.controller = Controller("ctrl", apps=[SyncCheckApp()])
    timing = TimingModeSync(t_ext=4, t_int=1)
    net = QuantumNetwork(topo, timing=timing)

    s = Simulator(0.0, 29.9, accuracy=1000, install_to=(net,))
    assert timing.t_ext.time_slot == 4000
    assert timing.t_rtg.time_slot == 0
    assert timing.t_int.time_slot == 1000

    s.run()

    for node in net.nodes + [net.get_controller()]:
        app = node.get_app(SyncCheckApp)
        assert app.enters == 12
        assert app.exits == 11
