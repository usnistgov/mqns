import itertools
from dataclasses import dataclass

import pytest

from mqns.entity.base_channel import default_light_speed
from mqns.entity.memory import QuantumMemory
from mqns.entity.node import QNode
from mqns.entity.qchannel import LinkArch, LinkArchDimBk, LinkArchDimBkSeq, LinkArchDimDual, LinkArchSim, LinkArchSr
from mqns.models.delay import ConstantDelayModel, DelayModel
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.models.error import DepolarErrorModel, ErrorModel, PerfectErrorModel, make_time_decay_func
from mqns.simulator import Simulator, Time


@dataclass
class FakeQuantumChannel:
    length: float
    alpha: float
    delay: DelayModel
    transfer_error: ErrorModel

    def __init__(self, length: float, *, alpha=0.2, delay=-1.0, transfer_error_rate=0.0):
        self.length = length
        self.alpha = alpha
        self.delay = ConstantDelayModel(delay if delay >= 0 else length / default_light_speed[0])
        self.transfer_error = DepolarErrorModel().set(rate=transfer_error_rate, length=0)


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
    # attempt_duration = tml*tau_l + tm0*tau_0
    # notify_a = aml*tau_l + am0*tau_0
    # notify_b = bml*tau_l + bm0*tau_0
    rml, rm0, aml, am0, bml, bm0 = multipliers

    tau_l, tau_0 = 0.000471, 0.000031

    ch = FakeQuantumChannel(0, delay=tau_l, transfer_error_rate=0)
    link_arch = LA()
    link_arch.set(
        ch=ch,
        eta_s=1,
        eta_d=1,
        reset_time=0,
        tau_0=tau_0,
        epr_type=WernerStateEntanglement,
        init_fidelity=1.0,
    )

    d1_epr_creation, d1_notify_a, d1_notify_b = link_arch.delays(1)
    assert d1_epr_creation == pytest.approx(0.0, abs=1e-6)
    assert d1_notify_a == pytest.approx(tau_l * aml + tau_0 * am0, abs=1e-6)
    assert d1_notify_b == pytest.approx(tau_l * bml + tau_0 * bm0, abs=1e-6)

    d6_epr_creation, _, _ = link_arch.delays(6)
    assert d6_epr_creation - d1_epr_creation == pytest.approx((tau_l * rml + tau_0 * rm0) * 5, abs=1e-6)


ACCURACY = 10_000_000
EPR_TIME = Time(10, accuracy=ACCURACY)


def make_epr(link_arch: LinkArch, t_cohere: Time):
    src, dst = QNode("S"), QNode("D")
    for node in src, dst:
        node.memory = QuantumMemory("M", capacity=1, t_cohere=t_cohere.sec)

    _ = Simulator(accuracy=ACCURACY, install_to=(src, dst))

    epr, d_notify_a, d_notify_b = link_arch.make_epr(1, EPR_TIME, key="K", src=src, dst=dst)
    assert (epr.key, epr.src, epr.dst) == ("K", src, dst)
    return epr, d_notify_a, d_notify_b


@pytest.mark.parametrize(
    ("LA", "E"),
    itertools.product(
        [LinkArchDimBk, LinkArchDimBkSeq, LinkArchDimDual, LinkArchSr],
        [WernerStateEntanglement, MixedStateEntanglement],
    ),
)
def test_perfect_error(LA: type[LinkArch], E: type[Entanglement]):
    ch = FakeQuantumChannel(0)
    t_cohere = Time.from_sec(1, accuracy=ACCURACY)
    store_decay = make_time_decay_func(PerfectErrorModel(), t_cohere=t_cohere)
    link_arch = LA()
    link_arch.set(
        ch=ch,
        eta_s=1,
        eta_d=1,
        reset_time=0,
        tau_0=0,
        epr_type=E,
        t0=t_cohere,
        store_decays=(store_decay, store_decay),
        bsa_error={"p_error": 0},
    )

    epr, _, _ = make_epr(link_arch, t_cohere)
    assert type(epr) is E
    assert epr.fidelity_time == EPR_TIME
    assert epr.fidelity == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("LA", "w_or_probv"),
    [
        (LinkArchDimBk, 0.877971),
        (LinkArchDimBk, (0.880156, 0.045599, 0.037122, 0.037122)),
        (LinkArchDimBkSeq, 0.869235),
        (LinkArchDimBkSeq, (0.871841, 0.053913, 0.037122, 0.037122)),
        (LinkArchDimDual, 0.937001),
        (LinkArchDimDual, (0.937468, 0.023918, 0.019306, 0.019306)),
        (LinkArchSr, 0.982592),
        (LinkArchSr, (0.982636, 0.010696, 0.003333, 0.003333)),
        (LinkArchSim, 0.927650),
        (LinkArchSim, (0.928317, 0.026917, 0.022382, 0.022382)),
    ],
)
def test_realistic_error(LA: type[LinkArch], w_or_probv: float | tuple[float, float, float, float]):
    ch = FakeQuantumChannel(
        50.0,  # km
        transfer_error_rate=0.001,  # 0.001 for typical fiber, 0.0051 for noisy fiber
    )
    t_cohere = Time.from_sec(0.100, accuracy=ACCURACY)  # coherence of an NV-center or Ion-Trap
    store_decay = make_time_decay_func(t_cohere=t_cohere)
    link_arch = LA()
    link_arch.set(
        ch=ch,
        eta_s=1,
        eta_d=1,
        reset_time=0,
        tau_0=0.000001,  # 1~10us
        epr_type=MixedStateEntanglement if isinstance(w_or_probv, tuple) else WernerStateEntanglement,
        t0=Time(0, accuracy=t_cohere.accuracy),
        store_decays=(store_decay, store_decay),
        bsa_error={"p_error": 0.01},  # 0.5~2.0% detector jitter and beam-splitter asymmetry
    )

    epr, d_notify_a, d_notify_b = make_epr(link_arch, t_cohere)
    assert EPR_TIME <= epr.fidelity_time <= min(d_notify_a, d_notify_b)
    epr.apply_store_decays(now=max(d_notify_a, d_notify_b))
    print(epr)
    if type(epr) is WernerStateEntanglement:
        assert epr.w == pytest.approx(w_or_probv, abs=1e-6)
    elif type(epr) is MixedStateEntanglement:
        assert epr.probv == pytest.approx(w_or_probv, abs=1e-6)
