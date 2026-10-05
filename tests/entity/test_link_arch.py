from collections.abc import Sequence

import pytest

from mqns.entity.base_channel import default_light_speed
from mqns.entity.memory import QuantumMemory
from mqns.entity.node import QNode
from mqns.entity.qchannel import LinkArch, LinkArchDimBk, LinkArchDimBkSeq, LinkArchDimDual, LinkArchSim, LinkArchSr
from mqns.models.delay import ConstantDelayModel, DelayModel
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.models.error import DepolarErrorModel, ErrorModel, parse_time_decay, time_decay_nop
from mqns.simulator import Simulator, Time

ACCURACY = 100_000_000
EPR_TIME = Time(10, accuracy=ACCURACY)


class FakeQuantumChannel:
    length: float
    alpha: float
    eta_s: float = 1
    eta_d: float = 1
    delay: DelayModel
    init_fidelity: float | Sequence[float] | None
    transfer_error: ErrorModel
    bsa_error: ErrorModel

    def __init__(
        self,
        length: float,
        *,
        alpha=0.2,
        delay=-1.0,
        init_fidelity: float | Sequence[float] | None = None,
        transfer_error_rate=0.0,
        bsa_error_prob=0.0,
    ):
        self.length = length
        self.alpha = alpha
        self.delay = ConstantDelayModel(delay if delay >= 0 else length / default_light_speed[0])
        self.init_fidelity = init_fidelity
        self.transfer_error = DepolarErrorModel().set(rate=transfer_error_rate, length=0)
        self.bsa_error = DepolarErrorModel().set(p_error=bsa_error_prob)


@pytest.mark.parametrize(
    ("LA", "multipliers"),
    [
        (LinkArchDimBk, (2, 2, 2, 2, 2, 2)),
        (LinkArchDimBkSeq, (5, 2, 4, 2, 4, 2)),
        (LinkArchDimDual, (1, 1, 1, 1, 1, 1)),
        (LinkArchSr, (2, 1, 1, 1, 2, 1)),
        (LinkArchSim, (1, 1, 1, 1, 1, 1)),
    ],
)
def test_delays(LA: type[LinkArch], multipliers: tuple[float, float, float, float, float, float]):
    tau_l, tau_0 = 0.000471, 0.000031

    ch = FakeQuantumChannel(0, delay=tau_l, transfer_error_rate=0)
    la = LA()
    la.set(
        time_accuracy=ACCURACY,
        ch=ch,
        reset_time=0,
        tau_0=tau_0,
        epr_type=WernerStateEntanglement,
    )

    rml, rm0, aml, am0, bml, bm0 = multipliers
    assert la.attempt_interval.sec == pytest.approx(tau_l * rml + tau_0 * rm0, abs=1e-6)
    assert la.d_notify_pri.sec == pytest.approx(tau_l * aml + tau_0 * am0, abs=1e-6)
    assert la.d_notify_2nd.sec == pytest.approx(tau_l * bml + tau_0 * bm0, abs=1e-6)


def make_epr(link_arch: LinkArch, t_cohere: Time):
    src, dst = QNode("S"), QNode("D")
    for node in src, dst:
        node.memory = QuantumMemory("M", capacity=1, t_cohere=t_cohere.sec)

    _ = Simulator(0, 10, accuracy=ACCURACY, install_to=(src, dst))

    epr = link_arch.make_epr(EPR_TIME, src, dst, key=None)
    assert epr.src is src
    assert epr.dst is dst
    return epr


@pytest.mark.parametrize(
    ("E", "use_probv"),
    [
        (WernerStateEntanglement, False),
        (MixedStateEntanglement, False),
        (MixedStateEntanglement, True),
    ],
)
def test_init_fidelity(E: type[Entanglement], use_probv: bool):
    ch = FakeQuantumChannel(0, init_fidelity=(70, 20, 5, 5) if use_probv else 0.7)
    t_cohere = Time.from_sec(1, accuracy=ACCURACY)
    la = LinkArchDimDual()
    la.set(
        time_accuracy=ACCURACY,
        ch=ch,
        reset_time=0,
        tau_0=0,
        epr_type=E,
    )

    epr = make_epr(la, t_cohere)
    assert epr.fidelity_time == EPR_TIME
    assert epr.fidelity == pytest.approx(0.7, abs=1e-6)
    if type(epr) is MixedStateEntanglement:
        if use_probv:
            assert epr.probv[1] == pytest.approx(0.2, abs=1e-6)
        else:
            assert epr.probv[1] == pytest.approx(0.1, abs=1e-6)


@pytest.mark.parametrize("LA", [LinkArchDimBk, LinkArchDimBkSeq, LinkArchDimDual, LinkArchSr])
@pytest.mark.parametrize("E", [WernerStateEntanglement, MixedStateEntanglement])
def test_perfect_error(LA: type[LinkArch], E: type[Entanglement]):
    ch = FakeQuantumChannel(0)
    t_cohere = Time.from_sec(1, accuracy=ACCURACY)
    la = LA()
    la.set(
        time_accuracy=ACCURACY,
        ch=ch,
        reset_time=0,
        tau_0=0,
        epr_type=E,
        store_decays=(time_decay_nop, time_decay_nop),
    )

    epr = make_epr(la, t_cohere)
    assert type(epr) is E
    assert epr.fidelity_time == EPR_TIME
    assert epr.fidelity == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("LA", "w_or_probv"),
    [
        (LinkArchDimBk, 0.877971),
        (LinkArchDimBk, (0.910693, 0.032721, 0.028292, 0.028292)),
        (LinkArchDimBkSeq, 0.869235),
        (LinkArchDimBkSeq, (0.906325, 0.037089, 0.028292, 0.028292)),
        (LinkArchDimDual, 0.937001),
        (LinkArchDimDual, (0.953930, 0.016928, 0.014570, 0.014570)),
        (LinkArchSr, 0.982592),
        (LinkArchSr, (0.988796, 0.006203, 0.002500, 0.002500)),
        (LinkArchSim, 0.927650),
        (LinkArchSim, (0.946900, 0.019249, 0.016925, 0.016925)),
    ],
)
def test_realistic_error(LA: type[LinkArch], w_or_probv: float | tuple[float, float, float, float]):
    ch = FakeQuantumChannel(
        50.0,  # km
        transfer_error_rate=0.001,  # 0.001 for typical fiber, 0.0051 for noisy fiber
        bsa_error_prob=0.01,  # 0.5~2.0% detector jitter and beam-splitter asymmetry
    )
    t_cohere = Time.from_sec(0.100, accuracy=ACCURACY)  # coherence of an NV-center or Ion-Trap
    store_decay = parse_time_decay(None, t_cohere)
    la = LA()
    la.set(
        time_accuracy=ACCURACY,
        ch=ch,
        reset_time=0,
        tau_0=0.000001,  # 1~10us
        epr_type=MixedStateEntanglement if isinstance(w_or_probv, tuple) else WernerStateEntanglement,
        store_decays=(store_decay, store_decay),
    )

    epr = make_epr(la, t_cohere)
    assert EPR_TIME <= epr.fidelity_time <= EPR_TIME + min(la.d_notify_pri, la.d_notify_2nd)
    epr.apply_store_decays(now=EPR_TIME + max(la.d_notify_pri, la.d_notify_2nd))
    print(epr)
    if type(epr) is WernerStateEntanglement:
        assert epr.w == pytest.approx(w_or_probv, abs=1e-6)
    elif type(epr) is MixedStateEntanglement:
        assert epr.probv == pytest.approx(w_or_probv, abs=1e-6)
