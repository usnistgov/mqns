"""
Test suite for LinkLayer focused on timing within a ``LLGroup``.
"""

import numpy as np
import pytest

from mqns.entity.memory import QuantumMemory, QubitState
from mqns.entity.qchannel import LinkArchAlways, LinkArchDimBk, LinkArchSr
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.network.ll import LinkLayerCounters
from mqns.network.network import TimingModeSync
from mqns.simulator import Simulator, func_to_event
from mqns.utils import log

from .ll_common import activate_group, check_link_arch_delays, force_attempts, make_linear


@pytest.mark.parametrize("epr_type", [WernerStateEntanglement, MixedStateEntanglement])
def test_basic(monkeypatch: pytest.MonkeyPatch, epr_type: type[Entanglement]):
    net, (ch,), (ll1, ll2), (nl1, nl2) = make_linear(
        qchannel_args={"delay": 0.1, "link_arch": LinkArchDimBk()},
        memory_args={"t_cohere": 4.1},
        epr_type=epr_type,
    )
    simulator = Simulator(0.0, 20.0, install_to=(log, net))

    activate_group(0.5, 9.4, nl1, nl2, gid=7)
    check_link_arch_delays(0.5, ch, attempt_interval=0.2, d_notify_pri=0.2, d_notify_2nd=0.2)
    force_attempts(monkeypatch, ll1, n2=[1, 2, 1])

    nl1.check_epr_creation_prior = 0.2
    nl2.check_epr_creation_prior = 0.2
    nl1.release_after.append(2.9)
    nl2.release_after.append(3.2)

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        print(ll.node.name, ll.cnt, nl.entangle, nl.decohere)
        assert len(nl.entangle) == 3
        assert len(nl.decohere) == 2
        # t=0.5, path is installed with n1 as primary and n2 as secondary
        # t=0.5, n1 sends RESERVE_REQ
        # t=0.6, n2 receives RESERVE_REQ and sends RESERVE_RES
        # t=0.7, n1 receives RESERVE_RES, 1st attempt starts
        # t=0.9, entanglement established after 1 attempt, creation time was t=0.7
        assert nl.entangle[0] == pytest.approx(0.9, abs=1e-6)
        # t=3.8, n1 releases qubit and sends RESERVE_REQ
        # t=3.9, n2 receives RESERVE_REQ but has no qubit available
        # t=4.1, n2 releases qubit and sends RESERVE_RES
        # t=4.2, n1 receives RESERVE_RES, 1st attempt starts
        # t=4.4, 1st attempt fails, 2nd attempt starts
        # t=4.6, entanglement established after 2 attempts, creation time was t=4.4
        assert nl.entangle[1] == pytest.approx(4.6, abs=1e-6)
        # t=8.5, qubits decohered 4.1 seconds since entanglement creation
        assert nl.decohere[0] == pytest.approx(8.5, abs=1e-6)
        # t=8.5, n1 sends RESERVE_REQ
        # t=8.6, n2 receives RESERVE_REQ and sends RESERVE_RES
        # t=8.7, n1 receives RESERVE_RES, 1st attempt starts
        # t=8.9, entanglement established after 1 attempt, creation time was t=8.7
        assert nl.entangle[2] == pytest.approx(8.9, abs=1e-6)
        # t=9.4, path is deleted
        # t=12.8, qubits decohered 4.1 seconds since entanglement creation
        assert nl.decohere[1] == pytest.approx(12.8, abs=1e-6)
        # no more entanglements because the path has been deleted

    ll_cnt_agg = LinkLayerCounters.aggregate(net.nodes)
    print("ll_cnt_agg", ll_cnt_agg)
    for cnt in ll_cnt_agg, ll1.cnt:  # counters are only incremented on primary node
        assert cnt.n_etg == 3
        assert cnt.n_attempts == 4
        assert cnt.n_decoh == 2
        assert np.isnan(cnt.etg_rate)
        assert cnt.decoh_ratio == pytest.approx(2 / 3, abs=1e-6)

    assert ll2.cnt.n_etg == 0
    assert ll2.cnt.n_attempts == 0
    assert ll2.cnt.n_decoh == 0
    assert np.isnan(ll2.cnt.etg_rate)
    assert ll2.cnt.decoh_ratio == pytest.approx(0.0, abs=1e-6)

    QuantumMemory.check_leaks(net.nodes)


def test_skip_ahead():
    net, _, (ll1, ll2), (nl1, nl2) = make_linear(
        qchannel_args={"length": 100},
        memory_args={"t_cohere": 1.0},
    )
    simulator = Simulator(0.0, 10.0, install_to=(log, net))

    activate_group(0.5, None, nl1, nl2, gid=7)

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        print(ll.node.name, ll.cnt, nl.entangle, nl.decohere)

    assert len(nl1.entangle) == len(nl2.entangle) > 0
    for t1, t2 in zip(nl1.entangle, nl2.entangle, strict=True):
        assert t1 == pytest.approx(t2, abs=1e-6)

    ll_cnt_agg = LinkLayerCounters.aggregate(net.nodes)
    print("ll_cnt_agg", ll_cnt_agg)
    assert 0 <= ll_cnt_agg.decoh_ratio <= 1


@pytest.mark.parametrize(
    ("qchannel_delay", "t_entangle"),
    [
        # τ=0.2 for the channel between n0 and n1.
        # Entanglement (including reservation) requires 4τ i.e. 0.8 seconds but the EXTERNAL phase
        # has only 0.6 seconds, so that no entanglement could complete on this channel.
        (0.2, []),
        # τ=0.1 for the channel between n2 and n3.
        # Entanglement (including reservation) requires 4τ i.e. 0.4 seconds.
        # No entanglement occurs in the first EXTERNAL phase window, because reservations are only initiated
        # at the start of each EXTERNAL phase window, not when PathActivateEvent arrives.
        # The PathDeactivateEvent takes effect for the EXTERNAL phase window starting at t=6.0.
        (0.1, [1.4, 2.4, 3.4, 4.4, 5.4]),
    ],
)
def test_timing_sync(qchannel_delay: float, t_entangle: list[float]):
    net, _, _, (nl1, nl2) = make_linear(
        qchannel_args={"delay": qchannel_delay, "link_arch": LinkArchAlways(LinkArchDimBk())},
        memory_args={"t_cohere": 10.0},
        timing=TimingModeSync(t_ext=0.6, t_int=0.4),
    )
    simulator = Simulator(0.0, 6.1, install_to=(log, net))

    activate_group(0.1, 5.9, nl1, nl2, gid=7)
    simulator.run()

    for nl in nl1, nl2:
        print(nl.node.name, nl.entangle, nl.decohere)
        assert nl.entangle == pytest.approx(t_entangle, abs=1e-6)
        # All qubits are cleared at the start of each EXTERNAL phase, before memory decoherence occurs.
        # Decoherence events are not emitted for cleared qubits.
        assert len(nl.decohere) == 0

    QuantumMemory.check_leaks(net.nodes)


@pytest.mark.parametrize(
    ("t_delete", "entangled_at_delete", "n_entangle"),
    [
        # don't deactivate, only check timeline states
        (999, (0, 0), 3),
        # ---------------- first entanglement ----------------
        # deactivate when RESERVE_REQ is in-flight
        (0.2, (0, 0), 0),
        # deactivate when RESERVE_RES is in-flight
        (0.5, (0, 0), 0),
        # deactivate when waiting for t_notify_pri
        (0.8, (0, 0), 1),
        # deactivate when waiting for t_notify_2nd
        (1.1, (1, 0), 1),
        # deactivate after both qubits are delivered to NetworkLayer
        (1.5, (1, 1), 1),
        # ---------------- second entanglement ----------------
        # deactivate during first failed attempt
        (3.4, (0, 0), 1),
        # deactivate during second failed attempt
        (4.0, (0, 0), 1),
        # deactivate when the successful attempt starts
        (5.1, (0, 0), 2),
        # deactivate when nl2 owns a qubit
        (6.4, (0, 1), 2),
        # ---------------- third entanglement ----------------
        # deactivate when nl1 owns a qubit
        (8.2, (1, 0), 3),
    ],
)
def test_deactivate(
    monkeypatch: pytest.MonkeyPatch,
    t_delete: float,
    entangled_at_delete: tuple[int, int],
    n_entangle: int,
):
    """
    Verify ``LLDeactivateEvent`` that occurs in various steps.
    """
    net, (ch,), (ll1, ll2), (nl1, nl2) = make_linear(
        qchannel_args={"delay": 0.3, "link_arch": LinkArchSr()},
        memory_args={"t_cohere": 2.0},
    )
    simulator = Simulator(0.0, 100.0, install_to=(log, net))

    nl1.check_epr_creation_prior = 0.3
    nl2.check_epr_creation_prior = 0.6
    activate_group(0.1, t_delete, nl1, nl2, gid=7)
    check_link_arch_delays(0.1, ch, attempt_interval=0.6, d_notify_pri=0.3, d_notify_2nd=0.6)
    force_attempts(monkeypatch, ll1, n2=[1, 4, 1, 10000])

    def check_states(t: float, expected1: QubitState | None, expected2: QubitState | None) -> None:
        if t > t_delete:
            return

        def assert_states():
            if expected1 is not None:
                mq1, _ = nl1.memory.read(0)
                assert mq1.state is expected1
            if expected2 is not None:
                mq2, _ = nl2.memory.read(0)
                assert mq2.state is expected2

        event = func_to_event(simulator.time(sec=t), assert_states)
        event.priority = 10000
        simulator.sched(event)

    check_states(t_delete, *(QubitState.ENTANGLED if is_entangled else None for is_entangled in entangled_at_delete))

    expect_t_entangle1: list[float] = []
    expect_t_entangle2: list[float] = []
    # ---------------- first entanglement, k=1, decohere ----------------
    # t=0.1, activate with n1 as primary and n2 as secondary
    # t=0.1, n1 sets qubit state to ACTIVE, sends RESERVE_REQ
    check_states(0.1, QubitState.ACTIVE, QubitState.RAW)
    # t=0.4, n2 receives RESERVE_REQ, sets qubit state to RESERVED, sends RESERVE_RES
    check_states(0.4, QubitState.ACTIVE, QubitState.RESERVED)
    # t=0.7, n1 receives RESERVE_RES, sets qubit state to RESERVED; first attempt begins and it would succeed
    check_states(0.7, QubitState.RESERVED, QubitState.RESERVED)
    # t=1.0, n1 is notified, sets qubit state to ENTANGLED
    check_states(1.0, QubitState.ENTANGLED, QubitState.RESERVED)
    expect_t_entangle1.append(1.0)
    # t=1.3, n2 is notified, sets qubit state to ENTANGLED
    check_states(1.3, QubitState.ENTANGLED, QubitState.ENTANGLED)
    expect_t_entangle2.append(1.3)
    # t=2.7, EPR decoheres, qubit states are set to RELEASE then RAW
    nl1.release_after.append(None)
    nl2.release_after.append(None)
    #
    # ---------------- second entanglement, k=4, nl1 releases before nl2 ----------------
    # t=2.7, n1 sets qubit state to ACTIVE, sends RESERVE_REQ
    check_states(2.7, QubitState.ACTIVE, QubitState.RAW)
    # t=3.0, n2 receives RESERVE_REQ, sets qubit state to RESERVED, sends RESERVE_RES
    check_states(3.0, QubitState.ACTIVE, QubitState.RESERVED)
    # t=3.3, n1 receives RESERVE_RES, sets qubit state to RESERVED; first attempt begins and it would fail
    # t=3.9, first attempt fails, second attempt begins and it would fail
    # t=4.5, second attempt fails, third attempt begins and it would fail
    # t=5.1, third attempt fails, fourth attempt begins and it would succeed
    check_states(5.1, QubitState.RESERVED, QubitState.RESERVED)
    # t=5.4, n1 is notified, sets qubit state to ENTANGLED
    check_states(5.4, QubitState.ENTANGLED, QubitState.RESERVED)
    expect_t_entangle1.append(5.4)
    # t=5.7, n2 is notified, sets qubit state to ENTANGLED
    check_states(5.7, QubitState.ENTANGLED, QubitState.ENTANGLED)
    expect_t_entangle2.append(5.7)
    #
    # ---------------- third entanglement, k=1, nl2 releases before nl1 ----------------
    # t=6.3, n1 sets qubit state to RELEASE then RAW then ACTIVE, sends RESERVE_REQ
    nl1.release_after.append(6.3 - expect_t_entangle1[-1])
    check_states(6.3, QubitState.ACTIVE, QubitState.ENTANGLED)
    # t=6.5, n2 receives RESERVE_REQ but has no available qubit
    # t=6.8, n2 sets qubit state to RELEASE then RAW then RESERVED, sends RESERVE_RES
    nl2.release_after.append(6.8 - expect_t_entangle2[-1])
    check_states(6.8, QubitState.ACTIVE, QubitState.RESERVED)
    # t=7.1, n1 receives RESERVE_RES, sets qubit state to RESERVED; first attempt begins and it would succeed
    check_states(7.1, QubitState.RESERVED, QubitState.RESERVED)
    # t=7.4, n1 is notified, sets qubit state to ENTANGLED
    expect_t_entangle1.append(7.4)
    check_states(7.4, QubitState.ENTANGLED, QubitState.RESERVED)
    # t=7.7, n2 is notified, sets qubit state to ENTANGLED
    check_states(7.7, QubitState.ENTANGLED, QubitState.ENTANGLED)
    expect_t_entangle2.append(7.7)
    # t=8.1, n2 sets qubit state to RELEASE then RAW but has no pending request
    nl2.release_after.append(8.1 - expect_t_entangle2[-1])
    check_states(8.1, QubitState.ENTANGLED, QubitState.RAW)
    # t=8.5, n1 sets qubit state to RELEASE then RAW then ACTIVE, sends RESERVE_REQ
    nl1.release_after.append(8.5 - expect_t_entangle1[-1])
    # t=100, scenario ends

    simulator.run()

    if t_delete < simulator.te.sec:
        for ll in ll1, ll2:
            assert len(ll.entangle_by_peer) == 0
            assert len(ll.group_by_mq) == 0
    assert nl1.entangle == pytest.approx(expect_t_entangle1[:n_entangle], abs=1e-6)
    assert nl2.entangle == pytest.approx(expect_t_entangle2[:n_entangle], abs=1e-6)
