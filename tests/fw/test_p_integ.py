"""
Test suite for proactive forwarding integrated with LinkLayer.
"""

import pytest

from mqns.entity.memory import QuantumMemory
from mqns.entity.qchannel import LinkArch, LinkArchAlways, LinkArchDimBk, LinkArchSr
from mqns.entity.timer import Timer
from mqns.models.epr import EprTypeLiteral
from mqns.network.builder import NetworkBuilder
from mqns.network.fw import RoutingPath, RoutingPathInitArgs, SwapSequenceInput
from mqns.network.network import Request, RequestState, TimingMode, TimingModeAsync, TimingModeSync
from mqns.network.proactive import MuxSchemeInput, MuxSchemeStatistical, ProactiveForwarder
from mqns.network.protocol.consumer import RequestCounters
from mqns.network.protocol.link_layer import LinkLayer
from mqns.simulator import Time

from .fw_common import (
    LA_ALWAYS,
    build_grid_network,
    build_tree_network,
    collect_cpacket_counts,
    dflt_qchannel_args,
    make_network,
    print_node_counters,
)


@pytest.mark.parametrize("epr_type", ["W", "M"])
@pytest.mark.parametrize("timing", [TimingModeAsync(), TimingModeSync(t_ext=0.006, t_int=0.004)], ids=["ASYNC", "SYNC"])
@pytest.mark.parametrize("swap", ["asap", "l2r", "r2l"])
def test_4_swap(epr_type: EprTypeLiteral, timing: TimingMode, swap: SwapSequenceInput):
    """Test swapping in 4-node topology."""
    net, simulator = make_network(
        NetworkBuilder(epr_type=epr_type)
        .topo_linear(nodes="ABCD", t_cohere=1, ch_length=100, link_arch=LA_ALWAYS)
        .proactive_centralized(timing=timing),
        swap_table_leak_tol=256,
        has_link_layer=True,
        end_time=3.0,
    )
    _, fwB, fwC, _ = (node.get_app(ProactiveForwarder) for node in net.nodes)

    net.add_request(rp := RoutingPath("A", "D", swap=swap))
    simulator.run()
    print_node_counters(net)

    # The main purpose of this test is to verify that the forwarder can return released qubits back to LinkLayer
    # for re-generating elementary entanglements.
    # Hence, these numeric bounds are much smaller than usual values, but must be greater than the memory capacity.
    assert fwB.cnt.n_swapped >= 16
    assert fwC.cnt.n_swapped >= 16
    assert RequestCounters.of(net, rp).n_consumed >= 16


@pytest.mark.parametrize(
    ("mux", "swap", "end_time"),
    [
        pytest.param("B", "asap", 1, id="BufferSpace-asap"),
        pytest.param("B", "l2r", 1, id="BufferSpace-l2r"),
        pytest.param(MuxSchemeStatistical(select_swap_qubit="random"), "asap", 1, id="Statistical"),
        pytest.param("D", "asap", 10, id="DynamicEpr"),
    ],
)
@pytest.mark.parametrize("route_len", [3, 5])
@pytest.mark.parametrize("LA", [LinkArchDimBk, LinkArchSr])
def test_tree2_bidir(mux: MuxSchemeInput, swap: SwapSequenceInput, end_time: float, route_len: int, LA: type[LinkArch]):
    """Test bidirectional paths in tree topology."""
    net, simulator = build_tree_network(
        t_cohere=1.0,
        ch_capacity=2,
        qchannel_args=dflt_qchannel_args | {"link_arch": LinkArchAlways(LA())},
        fw={"p_swap": 1},
        mux=mux,
        swap_table_leak_tol=256,
        end_time=end_time,
        has_link_layer=True,
    )

    # Path 0 uses A-C or B-A-C segment in one direction.
    # Path 1 uses C-A or C-A-B segment in the opposite direction.
    rp_args = RoutingPathInitArgs(
        bufferspace_mv=1,
        swap=swap,
        swap_cutoff=[0.01, 0.01] * (route_len - 2),
    )
    net.add_request(req0 := Request(RoutingPath.static("DBACF"[-route_len:], **rp_args), active_period=(0.010, Time.MAX)))
    net.add_request(req1 := Request(RoutingPath.static("GCABE"[:route_len], **rp_args), active_period=(0.020, Time.MAX)))
    simulator.run()
    print_node_counters(net)

    consumed0 = RequestCounters.of(net, req0).n_consumed
    consumed1 = RequestCounters.of(net, req1).n_consumed

    if mux == "D" and route_len == 5:
        assert consumed0 + consumed1 > 0
        if min(consumed0, consumed1) == 0:
            pytest.xfail(reason="https://github.com/usnistgov/mqns/issues/60#issuecomment-5180421126")

    assert consumed0 > 0
    assert consumed1 > 0


def test_rect2_path_delete():
    """Test PATH_DELETE in 2x2 rectangle topology."""
    net, simulator = build_grid_network(t_cohere=1.0, has_link_layer=True)
    _, fwB, fwC, _ = (node.get_app(ProactiveForwarder) for node in net.nodes)
    llA, llB, llC, _ = (node.get_app(LinkLayer) for node in net.nodes)

    counters: list[tuple[int, int, int, int, int]] = []

    def save_counters():
        print_node_counters(net)
        counters.append(
            (
                fwB.cnt.n_swapped,
                llB.cnt.n_attempts,
                fwC.cnt.n_swapped,
                llC.cnt.n_attempts,
                llA.cnt.n_attempts,
            )
        )

    timer = Timer("save_counters", start_time=0.500, end_time=9.501, step_time=1.000, trigger_func=save_counters)
    timer.install(simulator)

    net.add_request(Request(RoutingPath.static("ABD"), active_period=(2, 6)))
    net.add_request(Request(RoutingPath.static("ACD"), active_period=(4, 8)))
    simulator.run()

    assert len(counters) == 10
    for i in 0, 1:  # fwB.cnt.n_swapped and llB.cnt.n_attempts
        assert counters[0][i] == counters[1][i]
        assert counters[6][i] == counters[9][i]
    for i in 2, 3:  # fwC.cnt.n_swapped and llC.cnt.n_attempts
        assert counters[0][i] == counters[3][i]
        assert counters[8][i] == counters[9][i]
    # llA.cnt.n_attempts
    assert counters[0][4] == counters[1][4]
    assert counters[8][4] == counters[9][4]

    QuantumMemory.check_leaks(net.nodes)


@pytest.mark.parametrize(
    ("k_paths", "n_requests"),
    [
        (-1, 3),
        (2, 1),
        # Currently allocation logic cannot accommodate multi-path and multi-request simultaneously.
    ],
)
@pytest.mark.parametrize("LA", [LinkArchDimBk, LinkArchSr])
def test_rect3_epr_count(monkeypatch: pytest.MonkeyPatch, k_paths: int, n_requests: int, LA: type[LinkArch]):
    """Test Request.epr_count in 3x3 rectangle topology."""
    net, simulator = build_grid_network(
        (3, 3),
        k_paths=k_paths,
        qchannel_args=dflt_qchannel_args | {"link_arch": LinkArchAlways(LA())},
        swap_table_leak_tol=256,
        has_link_layer=True,
    )

    requests: list[Request] = [
        Request("A-B", epr_count=7),
        Request("C-F", epr_count=5),
        Request("G-I", epr_count=3),
    ][:n_requests]
    net.add_request(*requests)

    cpacket_cnt = collect_cpacket_counts(monkeypatch, cp=True, cmd=True)
    simulator.run()
    print_node_counters(net)
    print("cpacket_cnt", cpacket_cnt)

    for req in requests:
        assert req.state is RequestState.EPR_COUNT_REACHED
        assert RequestCounters.of(net, req).n_consumed == req.epr_count
        for target in req.src, req.dst:
            assert cpacket_cnt[f"ctrl-{target}:PATH_INSERT"] == 1
            assert cpacket_cnt[f"ctrl-{target}:PATH_DELETE"] == 1
            assert cpacket_cnt[f"{target}-ctrl:PATH_REACH_EPR_COUNT"] == 1

    QuantumMemory.check_leaks(net.nodes)
