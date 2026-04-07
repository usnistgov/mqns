"""
Test suite for ReactiveForwarder focused on swapping.
"""

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, RecvClassicPacket, classic_cmd_handler
from mqns.entity.timer import Timer
from mqns.network.fw import RoutingController, RoutingPathStatic
from mqns.network.network import TimingModeSync
from mqns.network.reactive import ReactiveForwarder, ReactiveRoutingController
from mqns.network.reactive.message import LinkStateEntry, LinkStateMsg

from .fw_common import build_linear_network, build_tree_network, print_fw_counters, provide_entanglements


class ManualController(ClassicCommandDispatcherMixin, RoutingController):
    def __init__(self):
        super().__init__()
        self.add_handler(self.handle_classic_command, RecvClassicPacket)

        self.ls_pkts: list[tuple[ClassicPacket, LinkStateMsg]] = []
        self.ls_entries: list[LinkStateEntry] = []

    @classic_cmd_handler("LS")
    def handle_ls(self, pkt: ClassicPacket, msg: LinkStateMsg):
        _ = pkt
        self.ls_pkts.append((pkt, msg))
        self.ls_entries.extend(msg["ls"])


def test_tree2_one():
    """Verify link state messages and test one path in tree (height=2) topology."""
    ctrl = ManualController()
    net, simulator = build_tree_network(
        2,
        mode="R",
        ps=1.0,
        end_time=0.010,
        timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003),
        ctrl=ctrl,
    )
    f4 = net.get_node("n4").get_app(ReactiveForwarder)
    f2 = net.get_node("n2").get_app(ReactiveForwarder)
    f1 = net.get_node("n1").get_app(ReactiveForwarder)
    f3 = net.get_node("n3").get_app(ReactiveForwarder)
    f6 = net.get_node("n6").get_app(ReactiveForwarder)

    def do_routing():
        assert len(ctrl.ls_pkts) == 5
        assert len(ctrl.ls_entries) == 8
        ctrl.install_path(RoutingPathStatic(["n4", "n2", "n1", "n3", "n6"], swap=[2, 0, 1, 0, 2]))

    timer = Timer("do_routing", 0.0065, trigger_func=do_routing)
    timer.install(simulator)

    provide_entanglements(
        (0.0011, f4, f2),
        (0.0012, f2, f1),
        (0.0013, f1, f3),
        (0.0014, f3, f6),
    )
    simulator.run()
    print_fw_counters(net)

    assert f4.cnt.n_consumed == 1


def test_tree2_two():
    """Verify link state messages and test both paths in tree (height=2) topology."""
    ctrl = ManualController()
    net, simulator = build_tree_network(
        2,
        mode="R",
        qchannel_capacity=2,
        ps=1.0,
        end_time=0.010,
        timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003),
        ctrl=ctrl,
    )
    f4 = net.get_node("n4").get_app(ReactiveForwarder)
    f5 = net.get_node("n5").get_app(ReactiveForwarder)
    f2 = net.get_node("n2").get_app(ReactiveForwarder)
    f1 = net.get_node("n1").get_app(ReactiveForwarder)
    f3 = net.get_node("n3").get_app(ReactiveForwarder)
    f6 = net.get_node("n6").get_app(ReactiveForwarder)
    f7 = net.get_node("n7").get_app(ReactiveForwarder)

    def do_routing():
        assert len(ctrl.ls_pkts) == 7
        assert len(ctrl.ls_entries) == 16
        ctrl.install_path(RoutingPathStatic(["n4", "n2", "n1", "n3", "n6"], swap=[2, 0, 1, 0, 2], m_v=[(1, 1)] * 4))
        ctrl.install_path(RoutingPathStatic(["n5", "n2", "n1", "n3", "n7"], swap=[2, 0, 1, 0, 2], m_v=[(1, 1)] * 4))

    timer = Timer("do_routing", 0.0065, trigger_func=do_routing)
    timer.install(simulator)

    provide_entanglements(
        (0.0011, f4, f2),
        (0.0012, f2, f1),
        (0.0013, f1, f3),
        (0.0014, f3, f6),
        (0.0021, f5, f2),
        (0.0022, f2, f1),
        (0.0023, f1, f3),
        (0.0024, f3, f7),
    )
    simulator.run()
    print_fw_counters(net)

    assert f4.cnt.n_consumed == 1
    assert f5.cnt.n_consumed == 1


def test_3_minimal():
    """Test 3-node minimal swap."""
    net, simulator = build_linear_network(3, mode="R", ps=1.0, timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003))
    ctrl = net.get_controller().get_app(ReactiveRoutingController)
    f1 = net.get_node("n1").get_app(ReactiveForwarder)
    f2 = net.get_node("n2").get_app(ReactiveForwarder)
    f3 = net.get_node("n3").get_app(ReactiveForwarder)

    provide_entanglements(
        (1.001, f1, f2),
        (1.002, f2, f3),
    )
    simulator.run()
    print(ctrl.cnt)
    print_fw_counters(net)

    assert ctrl.cnt.n_ls == 3
    assert ctrl.cnt.n_decision == 1
    assert f1.cnt.n_consumed == 1
    assert f2.cnt.n_consumed == 0
    assert f3.cnt.n_consumed == 1
    assert f2.cnt.n_swapped == 1
