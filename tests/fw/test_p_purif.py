"""
Test suite for ProactiveForwarder focused on purification.
"""

import pytest

from mqns.network.fw import RoutingPathSingle
from mqns.network.proactive import ProactiveForwarder
from mqns.utils import rng

from .fw_common import build_linear_network, check_fw_counters, install_path, print_fw_counters, provide_entanglements


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
    n_etg: int = 2**n_rounds
    net, simulator = build_linear_network(2, qchannel_capacity=n_etg, fw={"p_swap": 0.0})
    fwA = net.get_node("A").get_app(ProactiveForwarder)
    fwB = net.get_node("B").get_app(ProactiveForwarder)

    install_path(net, RoutingPathSingle("A", "B", swap=[0, 0], purif={"A-B": n_rounds}))
    provide_entanglements(*((1.001 + i / 1000, fwA, fwB) for i in range(n_etg)))
    force_purify_outcome(monkeypatch, *(True if i > 0 else False for i in purif_success))
    simulator.run()
    print_fw_counters(net)

    assert fwA.cnt.n_purif == n_purif == fwB.cnt.n_purif
    n_eligible = 0 if len(n_purif) < n_rounds else n_purif[-1]
    check_fw_counters(
        net,
        n_entg=(n_etg, n_etg),
        n_eligible=(n_eligible, n_eligible),
        n_consumed=(n_eligible, n_eligible),
    )


def test_4_l2r(monkeypatch: pytest.MonkeyPatch):
    """Test multi-segment purification on 4-node topology with l2r swapping order."""
    net, simulator = build_linear_network(4, qchannel_capacity=8, fw={"p_swap": 1.0})
    fwA, fwB, fwC, fwD = (node.get_app(ProactiveForwarder) for node in net.nodes)

    install_path(
        net,
        RoutingPathSingle("A", "D", swap=[2, 0, 1, 2], purif={"A-B": 1, "B-C": 1, "C-D": 1, "A-C": 1, "A-D": 1}),
    )
    provide_entanglements(
        (1.001, fwA, fwB),  # \
        (1.002, fwA, fwB),  # -+ A-B purif_rounds=1 \
        (1.003, fwB, fwC),  # \                      \
        (1.004, fwB, fwC),  # -+ B-C purif_rounds=1 --+ A-C purif_rounds=0
        (1.005, fwA, fwB),  # \                             \
        (1.006, fwA, fwB),  # -+ A-B purif_rounds=1 \        + A-C purif_rounds=1
        (1.007, fwB, fwC),  # \                      \      /              \
        (1.008, fwB, fwC),  # -+ B-C purif_rounds=1 --+ A-C purif_rounds=0  + A-D purif_rounds=0
        (1.009, fwC, fwD),  # \                                            /      \
        (1.010, fwC, fwD),  # -+ C-D purif_rounds=1 ----------------------/        \
        (1.011, fwA, fwB),  # \                                                     |
        (1.012, fwA, fwB),  # -+ A-B purif_rounds=1 \                               |
        (1.013, fwB, fwC),  # \                      \                              +-- A-D purif_rounds=1
        (1.014, fwB, fwC),  # -+ B-C purif_rounds=1 --+ A-C purif_rounds=0          |
        (1.015, fwA, fwB),  # \                             \                       |
        (1.016, fwA, fwB),  # -+ A-B purif_rounds=1 \        + A-C purif_rounds=1  /
        (1.017, fwB, fwC),  # \                      \      /              \      /
        (1.018, fwB, fwC),  # -+ B-C purif_rounds=1 --+ A-C purif_rounds=0  + A-D purif_rounds=0
        (1.019, fwC, fwD),  # \                                            /
        (1.020, fwC, fwD),  # -+ C-D purif_rounds=1 ----------------------/
    )
    force_purify_outcome(monkeypatch, *[True] * 19)
    simulator.run()
    print_fw_counters(net)

    assert fwA.cnt.n_purif == [4 + 2 + 1]  # 4 with fwB, 2 with fwC, 1 with fwD
    assert fwB.cnt.n_purif == [4 + 4]  # 4 with fwA, 4 with fwC
    assert fwC.cnt.n_purif == [4 + 2 + 2]  # 4 with fwB, 2 with fwD, 2 with fwA
    assert fwD.cnt.n_purif == [2 + 1]  # 2 with fwC, 1 with fwA

    check_fw_counters(
        net,
        # An entanglement becomes eligible if it completes all purification and the node has lower/equal swap rank.
        # This differs from .cnt.n_purif[0], which does not consider the node's swap rank.
        # fwA: 1 with fwD
        # fwB: 4 with fwA, 4 with fwC
        # fwC: 2 with fwD, 2 with fwA
        # fwD: 1 with fwA
        n_eligible=(1, 4 + 4, 2 + 2, 1),
        n_consumed=(1, 0, 0, 1),
    )
