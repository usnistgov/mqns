"""
Test suite for ReactiveForwarder focused on swapping.
"""

from collections import defaultdict

import pytest

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, RecvClassicPacket, classic_cmd_handler
from mqns.network.fw import RoutingController, RoutingPathStatic
from mqns.network.network import TimingModeSync
from mqns.network.reactive import ReactiveForwarder, ReactiveRoutingController
from mqns.network.reactive.message import LinkStateEntry, LinkStateMsg
from mqns.simulator import func_to_event

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

    simulator.add_event(func_to_event(simulator.time(sec=0.0065), do_routing))

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

        qubits_by_channel = defaultdict[str, list[str]](lambda: [])
        for entry in ctrl.ls_entries:
            qubits_by_channel[f"{entry['node']}{entry['neighbor']}"].append(entry["qubit"])

        ctrl.install_path(
            RoutingPathStatic(
                ["n4", "n2", "n1", "n3", "n6"],
                swap=[2, 0, 1, 0, 2],
                m_v=[
                    qubits_by_channel["n4n2"].pop(),
                    qubits_by_channel["n2n1"].pop(),
                    qubits_by_channel["n1n3"].pop(),
                    qubits_by_channel["n3n6"].pop(),
                ],
            )
        )
        ctrl.install_path(
            RoutingPathStatic(
                ["n5", "n2", "n1", "n3", "n7"],
                swap=[2, 0, 1, 0, 2],
                m_v=[
                    qubits_by_channel["n5n2"].pop(),
                    qubits_by_channel["n2n1"].pop(),
                    qubits_by_channel["n1n3"].pop(),
                    qubits_by_channel["n3n7"].pop(),
                ],
            )
        )

    simulator.add_event(func_to_event(simulator.time(sec=0.0065), do_routing))

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


@pytest.mark.parametrize(
    ("req_active", "etg12", "etg23", "cnt"),
    [
        # Request is active in both slots, EPRs arrive in first slot, request satisfied.
        ((0, 0.020), [0.001], [0.002], (3, 1)),
        # Request is active in both slots, EPRs arrive in second slot, request satisfied.
        ((0, 0.020), [0.011], [0.012], (3, 1)),
        # Request is active in both slots, EPRs arrive in both slots, request satisfied twice.
        ((0, 0.020), [0.001, 0.011], [0.002, 0.012], (6, 2)),
        # Request is active in both slots, EPRs arrive in separate slots, request unsatisfied.
        ((0, 0.020), [0.001], [0.012], (4, 0)),
        # Request is active in first slot, EPRs arrive in second slot, request unsatisfied.
        ((0, 0.010), [0.011], [0.012], (3, 0)),
        # Request is active in first slot, EPRs arrive twice in first slot, request satisfied twice.
        ((0, 0.010), [0.001, 0.003], [0.002, 0.004], (3, 2)),
    ],
)
def test_3_minimal(req_active: tuple[float, float], etg12: list[float], etg23: list[float], cnt: tuple[int, int]):
    """Test 3-node minimal swap, two time slots."""
    net, simulator = build_linear_network(
        3,
        qchannel_capacity=2,
        mode="R",
        ps=1.0,
        end_time=0.020,
        timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003),
    )
    ctrl = net.get_controller().get_app(ReactiveRoutingController)
    f1 = net.get_node("n1").get_app(ReactiveForwarder)
    f2 = net.get_node("n2").get_app(ReactiveForwarder)
    f3 = net.get_node("n3").get_app(ReactiveForwarder)

    simulator.add_event(func_to_event(simulator.time(sec=req_active[0]), lambda: net.add_request(f1.node, f3.node)))
    simulator.add_event(func_to_event(simulator.time(sec=req_active[1]), net.requests.clear))
    provide_entanglements(
        *((t, f1, f2) for t in etg12),
        *((t, f2, f3) for t in etg23),
    )
    simulator.run()
    print(ctrl.cnt)
    print_fw_counters(net)

    assert (ctrl.cnt.n_ls, ctrl.cnt.n_satisfy) == cnt
    assert f1.cnt.n_consumed == cnt[1]
    assert f2.cnt.n_consumed == 0
    assert f3.cnt.n_consumed == cnt[1]
    assert f2.cnt.n_swapped == cnt[1]
