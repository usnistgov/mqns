"""
Test suite for ProactiveForwarder focused on purification.
"""

import pytest

from mqns.network.proactive import ProactiveForwarder, RoutingPathSingle
from mqns.utils import rng

from .proactive_common import (
    build_linear_network,
    install_path,
    print_fw_counters,
    provide_entanglements,
)


def force_purify_outcome(monkeypatch: pytest.MonkeyPatch, *success: bool):
    l = list(success)

    def new_random() -> float:
        nonlocal l
        this_success, *l = l
        return 0.0 if this_success else 1.0

    monkeypatch.setattr(rng, "random", new_random)


@pytest.mark.parametrize(
    ("n_rounds", "purif_success", "n_purif"),
    [
        # 1 round, success.
        (1, [1], [1]),
        # 1 round, failure.
        (1, [0], []),
        # 2 rounds, success.
        (2, [1] * 3, [2, 1]),
        # 2 rounds, round-0 success, round-1 failure.
        (2, [1, 1, 0], [2]),
        # 2 rounds, first round-0 success, second round-0 failure, round-1 not attempted.
        (2, [1, 0], [1]),
        # 2 rounds, first round-0 failure, second round-0 success, round-1 not attempted.
        (2, [0, 1], [1]),
        # 3 rounds, success.
        (3, [1] * 7, [4, 2, 1]),
        # 6 rounds, success.
        (6, [1] * 63, [32, 16, 8, 4, 2, 1]),
    ],
)
def test_link_rounds(monkeypatch: pytest.MonkeyPatch, n_rounds: int, purif_success: list[int], n_purif: list[int]):
    """Test multi-round purification on a single link with various purification outcomes."""
    n_etg = 2**n_rounds
    net, simulator = build_linear_network(2, ps=0.0, qchannel_capacity=n_etg)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)

    install_path(net, RoutingPathSingle("n1", "n2", swap=[0, 0], purif={"n1-n2": n_rounds}))
    provide_entanglements(*((1.001 + i / 1000, f1, f2) for i in range(n_etg)))
    force_purify_outcome(monkeypatch, *(True if i > 0 else False for i in purif_success))
    simulator.run()
    print_fw_counters(net)

    assert f1.cnt.n_entg == n_etg == f2.cnt.n_entg
    assert f1.cnt.n_purif == n_purif == f2.cnt.n_purif
    n_eligible = 0 if len(n_purif) < n_rounds else n_purif[-1]
    assert f1.cnt.n_eligible == n_eligible == f2.cnt.n_eligible
    assert f1.cnt.n_consumed == n_eligible == f2.cnt.n_consumed


def test_4_l2r(monkeypatch: pytest.MonkeyPatch):
    """Test multi-segment purification on 4-node topology with l2r swapping order."""
    net, simulator = build_linear_network(4, ps=1.0, qchannel_capacity=8)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    install_path(
        net,
        RoutingPathSingle("n1", "n4", swap=[2, 0, 1, 2], purif={"n1-n2": 1, "n2-n3": 1, "n3-n4": 1, "n1-n3": 1, "n1-n4": 1}),
    )
    provide_entanglements(
        (1.001, f1, f2),  # \
        (1.002, f1, f2),  # -+ n1-n2 purif_rounds=1 \
        (1.003, f2, f3),  # \                        \
        (1.004, f2, f3),  # -+ n2-n3 purif_rounds=1 --+ n1-n3 purif_rounds=0
        (1.005, f1, f2),  # \                                \
        (1.006, f1, f2),  # -+ n1-n2 purif_rounds=1 \         + n1-n3 purif_rounds=1
        (1.007, f2, f3),  # \                        \       /                \
        (1.008, f2, f3),  # -+ n2-n3 purif_rounds=1 --+ n1-n3 purif_rounds=0   + n1-n4 purif_rounds=0
        (1.009, f3, f4),  # \                                                 /     \
        (1.010, f3, f4),  # -+ n3-n4 purif_rounds=1 -------------------------/       \
        (1.011, f1, f2),  # \                                                         |
        (1.012, f1, f2),  # -+ n1-n2 purif_rounds=1 \                                 |
        (1.013, f2, f3),  # \                        \                                +-- n1-n4 purif_rounds=1
        (1.014, f2, f3),  # -+ n2-n3 purif_rounds=1 --+ n1-n3 purif_rounds=0          |
        (1.015, f1, f2),  # \                                \                        |
        (1.016, f1, f2),  # -+ n1-n2 purif_rounds=1 \         + n1-n3 purif_rounds=1  /
        (1.017, f2, f3),  # \                        \       /                \      /
        (1.018, f2, f3),  # -+ n2-n3 purif_rounds=1 --+ n1-n3 purif_rounds=0   + n1-n4 purif_rounds=0
        (1.019, f3, f4),  # \                                                 /
        (1.020, f3, f4),  # -+ n3-n4 purif_rounds=1 -------------------------/
    )
    force_purify_outcome(monkeypatch, *[True] * 19)
    simulator.run()
    print_fw_counters(net)

    assert f1.cnt.n_purif == [4 + 2 + 1]  # 4 with n2, 2 with n3, 1 with n4
    assert f2.cnt.n_purif == [4 + 4]  # 4 with n1, 4 with n3
    assert f3.cnt.n_purif == [4 + 2 + 2]  # 4 with n2, 2 with n4, 2 with n1
    assert f4.cnt.n_purif == [2 + 1]  # 2 with n3, 1 with n1

    # An entanglement becomes eligible if it completes all purification and the node has lower/equal swap rank.
    # This differs from .cnt.n_purif[0], which does not consider the node's swap rank.
    assert f1.cnt.n_eligible == 1  # 1 with n4
    assert f2.cnt.n_eligible == 4 + 4  # 4 with n1, 4 with n3
    assert f3.cnt.n_eligible == 2 + 2  # 2 with n4, 2 with n1
    assert f4.cnt.n_eligible == 1  # 1 with n1

    assert f1.cnt.n_consumed == 1 == f4.cnt.n_consumed
