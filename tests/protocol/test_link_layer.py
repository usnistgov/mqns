from collections import defaultdict, deque
from typing import override

import pytest

from mqns.entity.memory import MemoryDecohereEvent, MemoryQubit, QuantumMemory, QuantumMemoryInitKwargs, QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import LinkArchAlways, LinkArchDimBk, LinkArchSr, QuantumChannel, QuantumChannelInitKwargs
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.network.network import QuantumNetwork, TimingMode, TimingModeAsync, TimingModeSync
from mqns.network.protocol.event import PathActivateEvent, PathDeactivateEvent, QubitEntangledEvent, QubitReleasedEvent
from mqns.network.protocol.link_layer import LinkLayer, LinkLayerCounters, LinkLayerInitKwargs
from mqns.network.topology import ClassicTopology, LinearTopology
from mqns.simulator import Simulator, event_handler, func_to_event
from mqns.utils import log, rng


@pytest.fixture(autouse=True)
def _reseed_env():
    rng.reseed("env")


class NetworkLayer(Application[QNode]):
    check_epr_creation_prior: float | None = None
    """
    If set, ``handle_entangle`` verifies that every EPR was created this duration earlier.
    """

    def __init__(self):
        super().__init__()
        self.release_after = deque[float | None]()
        """If non-empty, ``QubitReleasedEvent`` would be emitted after specified duration for the next entanglement."""
        self.entangle: list[float] = []
        """Entanglement events, each entry is event time."""
        self.path_entangle = defaultdict[int | None, list[float]](list)
        """Entanglement times per path_id."""
        self.decohere: list[float] = []
        """Decoherence events, each entry is event time."""

    @override
    def install(self, node):
        self._application_install(node, QNode)
        self.memory = self.node.memory
        self.epr_type = self.node.network.epr_type

    @event_handler
    def handle_entangle(self, event: QubitEntangledEvent) -> None:
        mq, epr = self.memory.read(event.qubit.addr, has=self.epr_type)
        assert mq is event.qubit
        if self.check_epr_creation_prior is not None:
            t_create = epr.decohere_time - self.memory.t_cohere
            assert t_create == event.t - self.check_epr_creation_prior
        self.entangle.append(event.t.sec)
        self.path_entangle[mq.path_id].append(event.t.sec)

        try:
            release_after = self.release_after.popleft()
        except IndexError:
            return
        if release_after is not None:
            self.simulator.sched(func_to_event(event.t + release_after, self._release, mq))

    def _release(self, mq: MemoryQubit):
        self.memory.read(mq.addr, remove=True)
        mq.state = QubitState.RELEASE
        self.simulator.sched(QubitReleasedEvent(self.node, mq, t=self.simulator.tc))

    @event_handler
    def handle_decohere(self, event: MemoryDecohereEvent) -> None:
        self.decohere.append(event.t.sec)
        event.qubit.state = QubitState.RELEASE
        self.simulator.sched(QubitReleasedEvent(self.node, event.qubit, is_decoh=True, t=event.t))


def make_2nodes(
    *,
    ll_args: LinkLayerInitKwargs = {},
    qchannel_args: QuantumChannelInitKwargs,
    memory_args: QuantumMemoryInitKwargs,
    timing: TimingMode = TimingModeAsync(),
    epr_type: type[Entanglement] = WernerStateEntanglement,
) -> tuple[QuantumNetwork, QuantumChannel, LinkLayer, LinkLayer, NetworkLayer, NetworkLayer]:
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer(**ll_args)],
        qchannel_args=qchannel_args,
        cchannel_args="from_qchannel_args",
        memory_args=memory_args,
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow, timing=timing, epr_type=epr_type)
    net.build_route()
    ch = net.get_qchannel("n1", "n2")
    n1 = net.get_node("n1")
    n2 = net.get_node("n2")
    return net, ch, n1.get_app(LinkLayer), n2.get_app(LinkLayer), n1.get_app(NetworkLayer), n2.get_app(NetworkLayer)


def activate_path(
    t0: float | None, t1: float | None, src: Application[QNode], dst: Application[QNode], *, path_id: int | None, n=1
):
    """
    Schedule ``PathActivateEvent`` and ``PathDeactivateEvent``.
    """
    simulator = src.simulator
    ch = src.node.get_qchannel(dst.node)
    ch.assign_memory_qubits(capacity=n, path_id=path_id)

    if t0 is not None:
        t = simulator.time(sec=t0)
        simulator.sched(PathActivateEvent(src.node, ch, path_id, t=t, is_primary=True))
        simulator.sched(PathActivateEvent(dst.node, ch, path_id, t=t, is_primary=False))

    if t1 is not None:
        t = simulator.time(sec=t1)
        simulator.sched(PathDeactivateEvent(src.node, ch, path_id, t=t))
        simulator.sched(PathDeactivateEvent(dst.node, ch, path_id, t=t))


def check_link_arch_delays(
    t: float,
    ch: QuantumChannel,
    *,
    attempt_interval: float,
    d_notify_pri: float,
    d_notify_2nd: float,
) -> None:
    """
    Verify LinkArch computed delays match the expectation.
    """
    simulator = ch.simulator

    def assert_la_delays() -> None:
        la = ch.link_arch
        assert la.attempt_interval == simulator.time(sec=attempt_interval)
        assert la.d_notify_pri == simulator.time(sec=d_notify_pri)
        assert la.d_notify_2nd == simulator.time(sec=d_notify_2nd)

    event = func_to_event(simulator.time(sec=t), assert_la_delays)
    event.priority = 10000
    simulator.sched(event)


def force_attempts(monkeypatch: pytest.MonkeyPatch, ll: LinkLayer, /, **kwargs: list[int]):
    """
    Force entanglements to succeed at k-th attempt.

    Args:
        ll: LinkLayer instance.
        kwargs: Attempts numbers for upcoming entanglements, keyed by partner name.
    """

    def new_calc_attempts(qchannel: QuantumChannel) -> int:
        nonlocal kwargs
        partner = qchannel.find_peer(ll.node)
        l = kwargs[partner.name]
        this_attempts, *l = l
        kwargs[partner.name] = l
        return this_attempts

    monkeypatch.setattr(ll, "_calc_attempt", new_calc_attempts)


@pytest.mark.parametrize("epr_type", [WernerStateEntanglement, MixedStateEntanglement])
@pytest.mark.parametrize("path_id", [None, 7])
def test_basic(monkeypatch: pytest.MonkeyPatch, epr_type: type[Entanglement], path_id: int | None):
    net, ch, ll1, ll2, nl1, nl2 = make_2nodes(
        qchannel_args={"delay": 0.1, "link_arch": LinkArchDimBk()},
        memory_args={"t_cohere": 4.1},
        epr_type=epr_type,
    )
    simulator = Simulator(0.0, 20.0, install_to=(log, net))

    activate_path(0.5, 9.4, nl1, nl2, path_id=path_id)
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
    assert ll_cnt_agg.n_etg == 3
    assert ll_cnt_agg.n_attempts == 4
    # n_decoh is only incremented on the primary node.
    # Although there are two decoherence events in the simulation, n1.n_decoh is incremented only at t=8.3.
    # For the t=12.6 event, the path is deleted, so that n1 cannot recognize itself as primary.
    assert ll_cnt_agg.n_decoh == 1
    assert ll_cnt_agg.decoh_ratio == pytest.approx(1 / 3, abs=1e-6)

    QuantumMemory.check_leaks(net.nodes)


def test_multiple_paths():
    """
    Test multiple active paths.
    """
    net, _, ll1, ll2, nl1, nl2 = make_2nodes(
        qchannel_args={"delay": 0.005},
        memory_args={"capacity": 1 + 2 + 3 + 4, "t_cohere": 0.1},
    )
    simulator = Simulator(0.0, 10.0, install_to=(log, net))

    activate_path(0.1, 9.9, nl1, nl2, path_id=1, n=1)
    activate_path(0.1, 9.9, nl1, nl2, path_id=2, n=2)
    activate_path(0.1, 9.9, nl1, nl2, path_id=3, n=3)
    activate_path(0.1, 9.9, nl1, nl2, path_id=4, n=4)

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        path_entangle_cnts = [len(nl.path_entangle[path_id]) for path_id in range(1, 5)]
        print(ll.node.name, ll.cnt, f"path_entangle_cnts={path_entangle_cnts}")
        assert all(c > 0 for c in path_entangle_cnts)

    # All attempts should be initiated by ll1.
    assert ll1.cnt.n_attempts > 0
    assert ll2.cnt.n_attempts == 0


@pytest.mark.parametrize(
    ("t_delete", "entangled_at_delete", "n_entangle"),
    [
        # don't delete, only check timeline states
        (999, (0, 0), 3),
        # ---------------- first entanglement ----------------
        # path deleted when RESERVE_REQ is in-flight
        (0.2, (0, 0), 0),
        # path deleted when RESERVE_RES is in-flight
        (0.5, (0, 0), 0),
        # path deleted when waiting for t_notify_pri
        (0.8, (0, 0), 1),
        # path deleted when waiting for t_notify_2nd
        (1.1, (1, 0), 1),
        # path deleted when both qubits are owned by NetworkLayer
        (1.5, (1, 1), 1),
        # ---------------- second entanglement ----------------
        # path deleted during first failed attempt
        (3.4, (0, 0), 1),
        # path deleted during second failed attempt
        (4.0, (0, 0), 1),
        # path deleted when nl2 owns a qubit
        (6.4, (0, 1), 2),
        # ---------------- third entanglement ----------------
        # path deleted when nl1 owns a qubit
        (8.2, (1, 0), 3),
    ],
)
def test_path_delete(
    monkeypatch: pytest.MonkeyPatch,
    t_delete: float,
    entangled_at_delete: tuple[int, int],
    n_entangle: int,
):
    """
    Verify PATH_DELETE cleanup when deletion occurs in various steps.
    """
    net, ch, ll1, ll2, nl1, nl2 = make_2nodes(
        qchannel_args={"delay": 0.3, "link_arch": LinkArchSr()},
        memory_args={"t_cohere": 2.0},
    )
    simulator = Simulator(0.0, 100.0, install_to=(log, net))

    nl1.check_epr_creation_prior = 0.3
    nl2.check_epr_creation_prior = 0.6
    activate_path(0.1, t_delete, nl1, nl2, path_id=7)
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

    check_states(t_delete, *(QubitState.ENTANGLED1 if is_entangled else None for is_entangled in entangled_at_delete))

    expect_t_entangle1: list[float] = []
    expect_t_entangle2: list[float] = []
    # ---------------- first entanglement, k=1, decohere ----------------
    # t=0.1, path is installed with n1 as primary and n2 as secondary
    # t=0.1, n1 sets qubit state to ACTIVE, sends RESERVE_REQ
    check_states(0.1, QubitState.ACTIVE, QubitState.RAW)
    # t=0.4, n2 receives RESERVE_REQ, sets qubit state to RESERVED, sends RESERVE_RES
    check_states(0.4, QubitState.ACTIVE, QubitState.RESERVED)
    # t=0.7, n1 receives RESERVE_RES, sets qubit state to RESERVED; first attempt begins and it would succeed
    check_states(0.7, QubitState.RESERVED, QubitState.RESERVED)
    # t=1.0, n1 is notified, sets qubit state to ENTANGLED
    check_states(1.0, QubitState.ENTANGLED1, QubitState.RESERVED)
    expect_t_entangle1.append(1.0)
    # t=1.3, n2 is notified, sets qubit state to ENTANGLED
    check_states(1.3, QubitState.ENTANGLED1, QubitState.ENTANGLED1)
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
    check_states(5.3, QubitState.RESERVED, QubitState.RESERVED)
    # t=5.4, n1 is notified, sets qubit state to ENTANGLED
    check_states(5.4, QubitState.ENTANGLED1, QubitState.RESERVED)
    expect_t_entangle1.append(5.4)
    # t=5.7, n2 is notified, sets qubit state to ENTANGLED
    check_states(5.7, QubitState.ENTANGLED1, QubitState.ENTANGLED1)
    expect_t_entangle2.append(5.7)
    #
    # ---------------- third entanglement, k=1, nl2 releases before nl1 ----------------
    # t=6.3, n1 sets qubit state to RELEASE then RAW then ACTIVE, sends RESERVE_REQ
    nl1.release_after.append(6.3 - expect_t_entangle1[-1])
    check_states(6.3, QubitState.ACTIVE, QubitState.ENTANGLED1)
    # t=6.5, n2 receives RESERVE_REQ but has no available qubit
    # t=6.8, n2 sets qubit state to RELEASE then RAW then RESERVED, sends RESERVE_RES
    nl2.release_after.append(6.8 - expect_t_entangle2[-1])
    check_states(6.8, QubitState.ACTIVE, QubitState.RESERVED)
    # t=7.1, n1 receives RESERVE_RES, sets qubit state to RESERVED; first attempt begins and it would succeed
    check_states(7.1, QubitState.RESERVED, QubitState.RESERVED)
    # t=7.4, n1 is notified, sets qubit state to ENTANGLED
    expect_t_entangle1.append(7.4)
    check_states(7.4, QubitState.ENTANGLED1, QubitState.RESERVED)
    # t=7.7, n2 is notified, sets qubit state to ENTANGLED
    check_states(7.7, QubitState.ENTANGLED1, QubitState.ENTANGLED1)
    expect_t_entangle2.append(7.7)
    # t=8.1, n2 sets qubit state to RELEASE then RAW but has no pending request
    nl2.release_after.append(8.1 - expect_t_entangle2[-1])
    check_states(8.1, QubitState.ENTANGLED1, QubitState.RAW)
    # t=8.5, n1 sets qubit state to RELEASE then RAW then ACTIVE, sends RESERVE_REQ
    nl1.release_after.append(8.5 - expect_t_entangle1[-1])
    # t=100, scenario ends

    simulator.run()

    if t_delete < simulator.te.sec:
        assert len(ll1.channels) == 0
        assert len(ll2.channels) == 0
    assert nl1.entangle == pytest.approx(expect_t_entangle1[:n_entangle], abs=1e-6)
    assert nl2.entangle == pytest.approx(expect_t_entangle2[:n_entangle], abs=1e-6)


@pytest.mark.parametrize(
    ("t_delete", "nl_release", "t_insert", "path_id", "t_entangle1", "t_entangle2"),
    [
        # Stable timeline without path changes.
        #
        # ---------------- first entanglement, k=2 ----------------
        # t=0.1, initial path is inserted
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
        # Delete and reinsert when RESERVE_REQ is in-flight.
        # Rest of timeline is same as stable case delayed by 0.1s.
        pytest.param(0.2, [], 0.3, 7, [1.8, 5.0], [2.1, 5.3], id="RESERVE_REQ-same"),
        pytest.param(0.2, [], 0.3, 9, [1.8, 5.0], [2.1, 5.3], id="RESERVE_REQ-diff"),
        # Delete and reinsert when RESERVE_RES is in-flight.
        #
        # t=0.1, initial path is inserted
        # t=0.1, n1 requests reservation
        # t=0.4, n2 accepts reservation
        # t=0.5, initial path is deleted, n1 cancels reservation
        # t=0.6, replacement path is inserted, n1 requests reservation
        # t=0.8, n2 processes cancellation
        # t=0.9, n2 accepts reservation
        # Rest of timeline is same as stable case delayed by 0.5s.
        pytest.param(0.5, [], 0.6, 7, [2.1, 5.3], [2.4, 5.6], id="RESERVE_RES-same"),
        pytest.param(0.5, [], 0.6, 9, [2.1, 5.3], [2.4, 5.6], id="RESERVE_RES-diff"),
        # Delete and reinsert during the first (failing) attempt.
        # The first entanglement can no longer happen.
        # The second entanglement starts its reservation at reinsertion time.
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
        # Delete and reinsert during the second (successful) attempt, same path_id.
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
        # Delete and reinsert during the second (successful) attempt, different path_id.
        # The first entanglement is delivered per normal timeline.
        # The second entanglement is using separate qubits and starts its reservation at insertion time.
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
def test_path_reinsert(
    monkeypatch: pytest.MonkeyPatch,
    t_delete: float | None,
    nl_release: list[float],
    t_insert: float | None,
    path_id: int,
    t_entangle1: list[float],
    t_entangle2: list[float],
):
    """
    Test PATH_DELETE followed by PATH_INSERT.
    """
    same_path = path_id == 7
    net, ch, ll1, ll2, nl1, nl2 = make_2nodes(
        ll_args={"cnt_history": 2},
        qchannel_args={"delay": 0.3, "link_arch": LinkArchSr()},
        memory_args={"capacity": 1 if same_path else 2, "t_cohere": 2.0},
    )
    simulator = Simulator(0.0, 6.0, install_to=(log, net))

    activate_path(0.1, t_delete, nl1, nl2, path_id=7)
    activate_path(t_insert, None, nl1, nl2, path_id=path_id, n=0 if same_path else 1)
    check_link_arch_delays(0.1, ch, attempt_interval=0.6, d_notify_pri=0.3, d_notify_2nd=0.6)
    force_attempts(monkeypatch, ll1, n2=[2, 2, 10000])
    force_attempts(monkeypatch, ll2, n1=[2, 2, 10000])

    nl1.release_after += nl_release
    nl2.release_after += nl_release

    simulator.run()

    assert nl1.entangle == pytest.approx(t_entangle1, abs=1e-6)
    assert nl2.entangle == pytest.approx(t_entangle2, abs=1e-6)

    assert (cnt1 := ll1.cnt_channel(ch)) is not None
    # Given t_delete and t_insert are less than t_cohere apart, the per-channel counters would not
    # be deleted, so that they should include the entanglements from both activations.
    assert cnt1.n_etg == len(t_entangle1)
    # Each entanglement needs a total of 1.8 seconds: 0.6s reservation, 0.6s first attempt, 0.6s second attempt.
    # .etg_rate is rate per memory pair, so that it does not change even when there are two pairs.
    assert cnt1.etg_rate == pytest.approx(1 / 1.8, abs=1e-6)
    # Secondary node does not keep per-channel counters.
    assert ll2.cnt_channel(ch) is None


def test_skip_ahead():
    net, _, ll1, ll2, nl1, nl2 = make_2nodes(
        qchannel_args={"length": 100},
        memory_args={"t_cohere": 1.0},
    )
    simulator = Simulator(0.0, 10.0, install_to=(log, net))

    activate_path(0.5, None, nl1, nl2, path_id=None)

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
def test_timing_mode_sync(qchannel_delay: float, t_entangle: list[float]):
    net, _, _, _, nl1, nl2 = make_2nodes(
        qchannel_args={"delay": qchannel_delay, "link_arch": LinkArchAlways(LinkArchDimBk())},
        memory_args={"t_cohere": 10.0},
        timing=TimingModeSync(t_ext=0.6, t_int=0.4),
    )
    simulator = Simulator(0.0, 6.1, install_to=(log, net))

    activate_path(0.1, 5.9, nl1, nl2, path_id=None)  # insertion_count=1
    activate_path(1.1, 4.1, nl1, nl2, path_id=None, n=0)  # insertion_count=2

    simulator.run()

    for nl in nl1, nl2:
        print(nl.node.name, nl.entangle, nl.decohere)
        assert nl.entangle == pytest.approx(t_entangle, abs=1e-6)
        # All qubits are cleared at the start of each EXTERNAL phase, before memory decoherence occurs.
        # Decoherence events are not emitted for cleared qubits.
        assert len(nl.decohere) == 0

    QuantumMemory.check_leaks(net.nodes)
