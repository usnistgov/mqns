"""
Test suite for LinkLayer focused on having multiple groups.
"""

import pytest

from mqns.entity.qchannel import LinkArchSr
from mqns.simulator import Simulator
from mqns.utils import log

from .ll_common import activate_group, check_link_arch_delays, force_attempts, make_linear


@pytest.mark.parametrize(
    ("has_gid", "t0", "errmsg"),
    [
        pytest.param(None, None, "channel not activated", id="no-channel"),
        pytest.param(1, None, "does not have active group 7", id="no-group"),
        pytest.param(7, 0.2, "already has active group 7", id="duplicate-group"),
    ],
)
def test_error_gid(has_gid: int | None, t0: float | None, errmsg: str):
    """
    Test error conditions caused by absent/duplicate channel or group.
    """
    net, _, (ll1, ll2), (nl1, nl2) = make_linear(
        qchannel_args={"delay": 0.005},
        memory_args={"capacity": 2, "t_cohere": 0.1},
    )
    simulator = Simulator(0.0, 1.0, install_to=(log, net))

    if has_gid is not None:
        activate_group(0.1, None, nl1, nl2, gid=has_gid)
    activate_group(t0, 0.3, nl1, nl2, gid=7)

    with pytest.raises(RuntimeError, match=errmsg):
        simulator.run()


@pytest.mark.parametrize(
    ("roles1", "roles2", "errmsg"),
    [
        pytest.param((True, False), (False, True), "activated with opposite role", id="opposite"),
        pytest.param((True, True), None, "has same role on both ends", id="both-pri"),
        # Having two secondary nodes does not trigger an error.
        pytest.param((False, False), None, None, id="both-2nd"),
    ],
)
def test_error_roles(roles1: tuple[bool, bool], roles2: tuple[bool, bool] | None, errmsg: str | None):
    """
    Test error conditions caused by conflicting pri/2nd roles.
    """
    net, _, (ll1, ll2), (nl1, nl2) = make_linear(
        qchannel_args={"delay": 0.005},
        memory_args={"capacity": 2, "t_cohere": 0.1},
    )
    simulator = Simulator(0.0, 1.0, install_to=(log, net))

    activate_group(0.1, None, nl1, nl2, gid=1, roles=roles1)
    if roles2:
        activate_group(0.2, None, nl1, nl2, gid=2, roles=roles2)

    if errmsg is None:
        simulator.run()
        for cnt in ll1.cnt, ll2.cnt:
            assert cnt.n_attempts == 0
    else:
        with pytest.raises(RuntimeError, match=errmsg):
            simulator.run()


def test_multi_channel():
    """
    Test multiple active channels.
    """
    net, _, (ll1, ll2, ll3), (nl1, nl2, nl3) = make_linear(
        3,
        qchannel_args={"delay": 0.005},
        memory_args={"capacity": 2, "t_cohere": 0.1},
    )
    simulator = Simulator(0.0, 10.0, install_to=(log, net))

    activate_group(0.1, 9.9, nl1, nl2, gid=1)
    activate_group(0.1, 9.9, nl3, nl2, gid=1)

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2), (ll3, nl3):
        print(ll.node.name, ll.cnt, f"path_entangle_cnt={nl.group_entangle[1]}")
        assert len(nl.group_entangle[1]) > 0

    # All attempts should be initiated by ll1.
    assert ll1.cnt.n_attempts > 0
    assert ll2.cnt.n_attempts == 0
    assert ll3.cnt.n_attempts > 0


def test_multi_group():
    """
    Test multiple active groups on the same channel.
    """
    net, _, (ll1, ll2), (nl1, nl2) = make_linear(
        qchannel_args={"delay": 0.005},
        memory_args={"capacity": 1 + 2 + 3 + 4, "t_cohere": 0.1},
    )
    simulator = Simulator(0.0, 10.0, install_to=(log, net))

    activate_group(0.1, 9.9, nl1, nl2, gid=1, n=1)
    activate_group(0.1, 9.9, nl1, nl2, gid=2, n=2)
    activate_group(0.1, 9.9, nl1, nl2, gid=3, n=3)
    activate_group(0.1, 9.9, nl1, nl2, gid=4, n=4)

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        path_entangle_cnts = [len(nl.group_entangle[path_id]) for path_id in range(1, 5)]
        print(ll.node.name, ll.cnt, f"path_entangle_cnts={path_entangle_cnts}")
        assert all(c > 0 for c in path_entangle_cnts)

    # All attempts should be initiated by ll1.
    assert ll1.cnt.n_attempts > 0
    assert ll2.cnt.n_attempts == 0


@pytest.mark.parametrize(
    ("t_delete", "nl_release", "t_insert", "gid", "t_entangle1", "t_entangle2"),
    [
        # Stable timeline without path changes.
        #
        # ---------------- first entanglement, k=2 ----------------
        # t=0.1, gid=7 is activated
        # t=0.1, n1 requests reservation
        # t=0.4, n2 accepts reservation
        # t=0.7, first attempt begins
        # t=1.3, second attempt begins
        # t=1.6, n1 is notified
        # t=1.9, n2 is notified
        # t=3.3, qubits decohere
        #
        # ---------------- second entanglement, k=2 ----------------
        # t=3.3, n1 requests reservation
        # t=3.6, n2 accepts reservation
        # t=3.9, first attempt begins
        # t=4.5, second attempt begins
        # t=4.8, n1 is notified
        # t=5.1, n2 is notified
        # t=6.5, qubits decohere
        pytest.param(None, [], None, -1, [1.6, 4.8], [1.9, 5.1], id="stable"),
        # Deactivate and reactivate when RESERVE_REQ is in-flight.
        # Rest of timeline is same as stable case delayed by 0.1s.
        pytest.param(0.2, [], 0.3, 7, [1.8, 5.0], [2.1, 5.3], id="RESERVE_REQ-same"),
        pytest.param(0.2, [], 0.3, 9, [1.8, 5.0], [2.1, 5.3], id="RESERVE_REQ-diff"),
        # Deactivate and reactivate when RESERVE_RES is in-flight.
        #
        # t=0.1, gid=7 is activated
        # t=0.1, n1 requests reservation
        # t=0.4, n2 accepts reservation
        # t=0.5, gid=7 is deactivated, n1 cancels reservation
        # t=0.6, gid=7 or gid=9 is activated, n1 requests reservation
        # t=0.8, n2 processes cancellation
        # t=0.9, n2 accepts reservation
        # Rest of timeline is same as stable case delayed by 0.5s.
        pytest.param(0.5, [], 0.6, 7, [2.1, 5.3], [2.4, 5.6], id="RESERVE_RES-same"),
        pytest.param(0.5, [], 0.6, 9, [2.1, 5.3], [2.4, 5.6], id="RESERVE_RES-diff"),
        # Deactivate and reactivate during the first (failing) attempt.
        # The first entanglement can no longer happen.
        # The second entanglement starts its reservation at reactivation time.
        # ---------------- second entanglement, k=2 ----------------
        # t=0.9, n1 requests reservation
        # t=1.2, n2 accepts reservation
        # t=1.5, first attempt begins
        # t=2.1, second attempt begins
        # t=2.4, n1 is notified
        # t=2.7, n2 is notified
        # t=4.1, qubits decohere
        pytest.param(0.8, [], 0.9, 7, [2.4], [2.7], id="attempt1-same"),
        pytest.param(0.8, [], 0.9, 9, [2.4], [2.7], id="attempt1-diff"),
        # Deactivate and reactivate during the second (successful) attempt, same gid=7.
        # The first entanglement is delivered per normal timeline.
        # The second entanglement starts its reservation after qubits release.
        # ---------------- second entanglement, k=2 ----------------
        # t=1.7, n1 releases qubit, requests reservation
        # t=2.0, n2 releases qubit, accepts reservation
        # t=2.3, first attempt begins
        # t=2.9, second attempt begins
        # t=3.2, n1 is notified
        # t=3.5, n2 is notified
        # t=4.0, qubits decohere
        pytest.param(1.4, [0.1], 1.5, 7, [1.6, 3.2], [1.9, 3.5], id="attempt2-same"),
        # Deactivate and reactivate during the second (successful) attempt, different gid=9.
        # The first entanglement is delivered per normal timeline.
        # The second entanglement is using separate qubits and starts its reservation at reactivation time.
        # ---------------- second entanglement, k=2 ----------------
        # t=1.5, n1 requests reservation
        # t=1.8, n2 accepts reservation
        # t=2.1, first attempt begins
        # t=2.7, second attempt begins
        # t=3.0, n1 is notified
        # t=3.3, n2 is notified
        # t=3.8, qubits decohere
        pytest.param(1.4, [0.1], 1.5, 9, [1.6, 3.0], [1.9, 3.3], id="attempt2-diff"),
    ],
)
def test_reactivate(
    monkeypatch: pytest.MonkeyPatch,
    t_delete: float | None,
    nl_release: list[float],
    t_insert: float | None,
    gid: int,
    t_entangle1: list[float],
    t_entangle2: list[float],
):
    """
    Test deactivating and then re-activating a group on the same channel.
    """
    same_path = gid == 7  # TODO test capacity=1 even if gid is different
    net, (ch,), (ll1, ll2), (nl1, nl2) = make_linear(
        ll_args={"cnt_history": 2},
        qchannel_args={"delay": 0.3, "link_arch": LinkArchSr()},
        memory_args={"capacity": 1 if same_path else 2, "t_cohere": 2.0},
    )
    simulator = Simulator(0.0, 6.0, install_to=(log, net))

    activate_group(0.1, t_delete, nl1, nl2, gid=7)
    activate_group(t_insert, None, nl1, nl2, gid=gid)
    check_link_arch_delays(0.1, ch, attempt_interval=0.6, d_notify_pri=0.3, d_notify_2nd=0.6)
    force_attempts(monkeypatch, ll1, n2=[2, 2, 10000])
    force_attempts(monkeypatch, ll2, n1=[2, 2, 10000])

    nl1.release_after += nl_release
    nl2.release_after += nl_release

    simulator.run()

    assert nl1.entangle == pytest.approx(t_entangle1, abs=1e-6)
    assert nl2.entangle == pytest.approx(t_entangle2, abs=1e-6)

    assert (cnt1 := ll1.cnt_channel(ll2.node.name)) is not None
    # Given t_delete and t_insert are less than t_cohere apart, the per-channel counters would not
    # be deleted, so that they should include the entanglements from both activations.
    assert cnt1.n_etg == len(t_entangle1)
    # Each entanglement needs a total of 1.8 seconds: 0.6s reservation, 0.6s first attempt, 0.6s second attempt.
    # .etg_rate is rate per memory pair, so that it does not change even when there are two pairs.
    assert cnt1.etg_rate == pytest.approx(1 / 1.8, abs=1e-6)
    # Secondary node does not keep per-channel counters.
    assert ll2.cnt_channel(ll1.node.name) is None
