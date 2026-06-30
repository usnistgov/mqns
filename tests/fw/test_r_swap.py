"""
Test suite for ReactiveForwarder focused on swapping.
"""

from collections import defaultdict
from itertools import pairwise

import pytest

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, classic_cmd_handler
from mqns.network.fw import ForwarderConsumeCounters, RoutingController, RoutingPathStatic
from mqns.network.network import TimingModeSync
from mqns.network.reactive import ReactiveForwarder, ReactiveRoutingController
from mqns.network.reactive.message import LinkStateEntry, LinkStateMsg
from mqns.simulator import func_to_event

from .fw_common import (
    build_linear_network,
    build_tree_network,
    check_fw_counters,
    check_path_counters,
    print_fw_counters,
    provide_entanglements,
)


class ManualController(ClassicCommandDispatcherMixin, RoutingController):
    def __init__(self):
        super().__init__()

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
        fw={"p_swap": 1.0},
        end_time=0.010,
        timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003),
        ctrl=ctrl,
    )
    fwA, fwB, fwC, fwD, _, fwF, _ = (node.get_app(ReactiveForwarder) for node in net.nodes)

    def do_routing():
        assert len(ctrl.ls_pkts) == 5
        assert len(ctrl.ls_entries) == 8
        ctrl.install_path(RoutingPathStatic(["D", "B", "A", "C", "F"], swap=[2, 0, 1, 0, 2]))

    simulator.add_event(func_to_event(simulator.time(sec=0.0065), do_routing))

    provide_entanglements(
        (0.0011, fwD, fwB),
        (0.0012, fwB, fwA),
        (0.0013, fwA, fwC),
        (0.0014, fwC, fwF),
    )
    simulator.run()
    print_fw_counters(net)

    consume_cnt = ForwarderConsumeCounters.of_path(net, "D", "F")
    assert consume_cnt.n_consumed == 1


def test_tree2_two():
    """Verify link state messages and test both paths in tree (height=2) topology."""
    ctrl = ManualController()
    net, simulator = build_tree_network(
        2,
        mode="R",
        qchannel_capacity=2,
        fw={"p_swap": 1.0},
        end_time=0.010,
        timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003),
        ctrl=ctrl,
    )
    fwA, fwB, fwC, fwD, fwE, fwF, fwG = (node.get_app(ReactiveForwarder) for node in net.nodes)

    def do_routing():
        assert len(ctrl.ls_pkts) == 7
        assert len(ctrl.ls_entries) == 16

        qubits_by_channel = defaultdict[str, list[str]](lambda: [])
        for entry in ctrl.ls_entries:
            qubits_by_channel[f"{entry['node']}{entry['neighbor']}"].append(entry["qubit"])

        for route in "DBACF", "EBACG":
            ctrl.install_path(
                RoutingPathStatic(
                    route, swap=[2, 0, 1, 0, 2], m_v=[qubits_by_channel[f"{a}{b}"].pop() for a, b in pairwise(route)]
                )
            )

    simulator.add_event(func_to_event(simulator.time(sec=0.0065), do_routing))

    provide_entanglements(
        (0.0011, fwD, fwB),
        (0.0012, fwB, fwA),
        (0.0013, fwA, fwC),
        (0.0014, fwC, fwF),
        (0.0021, fwE, fwB),
        (0.0022, fwB, fwA),
        (0.0023, fwA, fwC),
        (0.0024, fwC, fwG),
    )
    simulator.run()
    print_fw_counters(net)

    consumeDF = ForwarderConsumeCounters.of_path(net, "D", "F")
    assert consumeDF.n_consumed == 1
    consumeEG = ForwarderConsumeCounters.of_path(net, "E", "G")
    assert consumeEG.n_consumed == 1


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
        fw={"p_swap": 1.0},
        end_time=0.020,
        timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003),
    )
    ctrl = net.get_controller().get_app(ReactiveRoutingController)
    fwA, fwB, fwC = (node.get_app(ReactiveForwarder) for node in net.nodes)

    simulator.add_event(func_to_event(simulator.time(sec=req_active[0]), lambda: net.add_request(fwA.node, fwC.node)))
    simulator.add_event(func_to_event(simulator.time(sec=req_active[1]), net.requests.clear))
    provide_entanglements(
        *((t, fwA, fwB) for t in etg12),
        *((t, fwB, fwC) for t in etg23),
    )
    simulator.run()
    print(ctrl.cnt)
    print_fw_counters(net)

    assert (ctrl.cnt.n_ls, ctrl.cnt.n_satisfy) == cnt
    check_fw_counters(
        net,
        n_swapped=(0, cnt[1], 0),
    )
    check_path_counters(net, n_consumed=cnt[1])
