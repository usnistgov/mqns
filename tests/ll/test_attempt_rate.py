import pytest

from mqns.entity.memory import QuantumMemory
from mqns.entity.qchannel import LinkArchDimBk
from mqns.network.ll import LinkLayer
from mqns.network.network import TimingModeSync
from mqns.simulator import Simulator
from mqns.utils import log

from .ll_common import activate_path, check_link_arch_delays, force_attempts, make_2nodes


@pytest.mark.parametrize(
    ("max_attempt_rate", "n_attempts", "expected"),
    [
        # Unlimited: both reservations attempt every 0.2s, success at t=0.7+0.2*2=1.1, notify +0.2.
        (None, [3, 3], [1.3, 1.3]),
        # 10/s (spacing 0.1) does not exceed link_arch attempt_interval even with 2 concurrent reservations.
        (10.0, [3, 3], [1.3, 1.3]),
        # 2/s (spacing 0.5): 2 concurrent reservations, interval max(0.2, 0.5*2)=1.0,
        # both succeed at t=0.7+1.0*2=2.7, notify 2.9.
        (2.0, [3, 3], [2.9, 2.9]),
        # 2/s: 2 concurrent reservations, interval 1.0.
        # 1st succeeds at t=0.7+1.0*1=1.7, notify 1.9.
        # 2nd has made 1 attempt by t=1.7, then becomes alone with interval 0.5,
        # remaining 2 attempts, succeeds at t=1.7+0.5*2=2.7, notify 2.9.
        (2.0, [2, 4], [1.9, 2.9]),
        # 2/s: 1st succeeds on its first attempt at t=0.7, then 2nd becomes alone with interval 0.5,
        # succeeds at t=0.7+0.5*2=1.7, notify 1.9.
        (2.0, [1, 3], [0.9, 1.9]),
    ],
)
def test_attempt_rate(
    monkeypatch: pytest.MonkeyPatch, max_attempt_rate: float | None, n_attempts: list[int], expected: list[float]
):
    net, ch, ll1, ll2, nl1, nl2 = make_2nodes(
        ll_args={} if max_attempt_rate is None else {"max_attempt_rate": max_attempt_rate},
        qchannel_args={"delay": 0.1, "link_arch": LinkArchDimBk()},
        memory_args={"capacity": 2, "t_cohere": 10.0},
    )
    simulator = Simulator(0.0, 5.0, install_to=(log, net))

    activate_path(0.5, None, nl1, nl2, path_id=None, n=2)
    check_link_arch_delays(0.5, ch, attempt_interval=0.2, d_notify_pri=0.2, d_notify_2nd=0.2)
    force_attempts(monkeypatch, ll1, n2=n_attempts)

    simulator.run()

    for nl in nl1, nl2:
        assert sorted(nl.entangle) == pytest.approx(expected, abs=1e-6)
    assert ll1.cnt.n_attempts == sum(n_attempts)


def test_attempt_rate_abort(monkeypatch: pytest.MonkeyPatch):
    """
    Aborting a reservation upon path deactivation allows remaining reservations to attempt more frequently.
    """
    net, _, ll1, _, nl1, nl2 = make_2nodes(
        ll_args={"max_attempt_rate": 2.0},
        qchannel_args={"delay": 0.1, "link_arch": LinkArchDimBk()},
        memory_args={"capacity": 2, "t_cohere": 10.0},
    )
    simulator = Simulator(0.0, 5.0, install_to=(log, net))

    activate_path(0.5, 1.2, nl1, nl2, path_id=1)
    activate_path(0.5, None, nl1, nl2, path_id=2)
    force_attempts(monkeypatch, ll1, n2=[3, 3])

    simulator.run()

    # Both reservations start at t=0.7 with interval 1.0.
    # At t=1.2, path 1 reservation is aborted; path 2 reservation has made 0.5 attempts.
    # Path 2 reservation becomes alone with interval 0.5, remaining 1.5 attempts,
    # succeeds at t=1.2+0.5*1.5=1.95, notify 2.15.
    for nl in nl1, nl2:
        assert nl.path_entangle[1] == []
        assert nl.path_entangle[2] == pytest.approx([2.15], abs=1e-6)
    assert ll1.cnt.n_attempts == 3


@pytest.mark.parametrize(
    ("n_attempts", "t_entangle_in_phase"),
    [
        # 1st succeeds on its first attempt at t=x.1, notify x.2.
        # 2nd becomes alone with interval max(0.1, 0.25)=0.25, succeeds at t=x.1+0.25=x.35, notify x.45.
        ([1, 2], [0.2, 0.45]),
        # 2nd would succeed at t=x.1+0.25*3=x.85, beyond EXTERNAL phase that ends at x.6.
        # Its pending throttling event must be canceled upon EXTERNAL phase exit.
        ([1, 4], [0.2]),
    ],
)
def test_attempt_rate_sync(monkeypatch: pytest.MonkeyPatch, n_attempts: list[int], t_entangle_in_phase: list[float]):
    net, ch, ll1, _, nl1, nl2 = make_2nodes(
        ll_args={"max_attempt_rate": 4.0},
        qchannel_args={"delay": 0.05, "link_arch": LinkArchDimBk()},
        memory_args={"capacity": 2, "t_cohere": 10.0},
        timing=TimingModeSync(t_ext=0.6, t_int=0.4),
    )
    simulator = Simulator(0.0, 6.1, install_to=(log, net))

    activate_path(0.1, 5.9, nl1, nl2, path_id=None, n=2)
    check_link_arch_delays(1.0, ch, attempt_interval=0.1, d_notify_pri=0.1, d_notify_2nd=0.1)
    force_attempts(monkeypatch, ll1, n2=n_attempts * 5)

    simulator.run()

    expected = [phase + t for phase in (1, 2, 3, 4, 5) for t in t_entangle_in_phase]
    for nl in nl1, nl2:
        assert sorted(nl.entangle) == pytest.approx(expected, abs=1e-6)

    QuantumMemory.check_leaks(net.nodes)


@pytest.mark.parametrize("max_attempt_rate", [0.0, -1.0])
def test_attempt_rate_invalid(max_attempt_rate: float):
    with pytest.raises(ValueError, match="max_attempt_rate must be positive"):
        LinkLayer(max_attempt_rate=max_attempt_rate)
