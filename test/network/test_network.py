from qns.entity import Application, Controller
from qns.network.network import QuantumNetwork, SignalTypeEnum, TimingModeEnum
from qns.network.topology import BasicTopology
from qns.simulator import Simulator


class SyncCheckApp(Application):
    def __init__(self):
        super().__init__()
        self.changes = 0

    def handle_sync_signal(self, signal_type: SignalTypeEnum):
        t = self.simulator.tc.sec
        assert (t % 5 == 0 and signal_type == SignalTypeEnum.EXTERNAL) or (
            t % 5 == 4 and signal_type == SignalTypeEnum.INTERNAL
        )
        self.changes += 1


def test_timing_mode_sync():
    topo = BasicTopology(2, nodes_apps=[SyncCheckApp()])
    topo.controller = Controller("ctrl", apps=[SyncCheckApp()])
    net = QuantumNetwork(topo=topo, timing_mode=TimingModeEnum.SYNC, t_ext=4, t_int=1)

    simulator = Simulator(start_second=0.0, end_second=29.9)
    net.install(simulator)
    simulator.run()

    for node in net.nodes + [net.get_controller()]:
        app = node.get_app(SyncCheckApp)
        assert app.changes == 12
