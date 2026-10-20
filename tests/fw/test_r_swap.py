"""
Test suite for reactive forwarding focused on swapping.
"""

import pytest

from mqns.network.builder import NetworkBuilder
from mqns.network.network import Request
from mqns.network.protocol.consumer import RequestCounters
from mqns.network.reactive import ReactiveForwarder, ReactiveRoutingController
from mqns.utils import unwrap

from .fw_common import (
    LA_ALWAYS,
    check_fw_counters,
    collect_cpacket_counts,
    make_network,
    print_node_counters,
    provide_entanglements,
)


@pytest.mark.parametrize(
    ("req_active", "etgAB", "etgBC", "cnt"),
    [
        # Request is active in both slots, EPRs arrive in first slot, request satisfied.
        ((0, 0.020), [0.001], [0.002], (3, 1, 1)),
        # Request is active in both slots, EPRs arrive in second slot, request satisfied.
        ((0, 0.020), [0.011], [0.012], (3, 1, 1)),
        # Request is active in both slots, EPRs arrive in both slots, request satisfied twice.
        ((0, 0.020), [0.001, 0.011], [0.002, 0.012], (6, 2, 2)),
        # Request is active in both slots, EPRs arrive in separate slots, request unsatisfied.
        ((0, 0.020), [0.001], [0.012], (4, 0, 0)),
        # Request is active in first slot, EPRs arrive in second slot, request unsatisfied.
        ((0, 0.010), [0.011], [0.012], (3, 0, 0)),
        # Request is active in first slot, EPRs arrive twice in first slot, request satisfied twice.
        ((0, 0.010), [0.001, 0.003], [0.002, 0.004], (3, 2, 1)),
    ],
)
def test_3_minimal(
    monkeypatch: pytest.MonkeyPatch,
    req_active: tuple[float, float],
    etgAB: list[float],
    etgBC: list[float],
    cnt: tuple[int, int, int],
):
    """Test 3-node minimal swap, two time slots."""
    net, simulator = make_network(
        NetworkBuilder()
        .topo_linear(nodes="ABC", t_cohere=1, ch_capacity=2, link_arch=LA_ALWAYS)
        .reactive_centralized(timing=(0.006, 0.001, 0.003), p_swap=1.0),
        end_time=0.020,
    )
    ctrl = unwrap(net.controller).get_app(ReactiveRoutingController)
    fwA, fwB, fwC = (node.get_app(ReactiveForwarder) for node in net.nodes)

    net.add_request(req := Request("A-C", active_period=req_active))
    provide_entanglements(
        *((t, fwA, fwB) for t in etgAB),
        *((t, fwB, fwC) for t in etgBC),
    )
    cpacket_cnt = collect_cpacket_counts(monkeypatch, cp=True, cmd=True)
    simulator.run()
    print(ctrl.cnt)
    print_node_counters(net)
    print("cpacket_cnt", cpacket_cnt)

    assert ctrl.cnt.n_ls == cnt[0]
    assert ctrl.cnt.n_satisfy == cnt[1]
    check_fw_counters(
        net,
        n_swapped=(0, cnt[1], 0),
    )
    assert RequestCounters.of(net, req).n_consumed == cnt[1]
    assert cpacket_cnt["*-ctrl:LS"] == cnt[0]
    assert cpacket_cnt["ctrl-*:PATH_INSERT"] == cnt[2] * 3
