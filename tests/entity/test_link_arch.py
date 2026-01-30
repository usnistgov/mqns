import pytest

from mqns.entity.qchannel import LinkArch, LinkArchDimBk, LinkArchDimBkSeq, LinkArchDimDual, LinkArchSim, LinkArchSr

type DurationMultipliers = tuple[float, float]
"""
Link architecture duration multiplier: (m_l, m_0).
The duration is: (m_l * tau_l + m_0 * tau_0).
"""


def check_link_arch(
    link_arch: LinkArch,
    *,
    attempt_duration: DurationMultipliers,
    epr_creation: DurationMultipliers = (0, 0),
    notify_a: DurationMultipliers,
    notify_b: DurationMultipliers,
):
    tau_l, tau_0 = 0.000471, 0.000031

    def compare_to(m: DurationMultipliers, m2=1.0):
        return pytest.approx((tau_l * m[0] + tau_0 * m[1]) * m2, abs=1e-6)

    d1_epr_creation, d1_notify_a, d1_notify_b = link_arch.delays(1, reset_time=0.0, tau_l=tau_l, tau_0=tau_0)
    assert d1_epr_creation == compare_to(epr_creation)
    assert d1_notify_a == compare_to(notify_a)
    assert d1_notify_b == compare_to(notify_b)

    d6_epr_creation, _, _ = link_arch.delays(6, reset_time=0.0, tau_l=tau_l, tau_0=tau_0)
    assert d6_epr_creation - d1_epr_creation == compare_to(attempt_duration, 5)


def test_dim_bk():
    link_arch = LinkArchDimBk()
    check_link_arch(link_arch, attempt_duration=(2, 2), notify_a=(2, 2), notify_b=(2, 2))


def test_dim_bk_seq():
    link_arch = LinkArchDimBkSeq()
    check_link_arch(link_arch, attempt_duration=(5, 2), notify_a=(4, 2), notify_b=(4, 2))


def test_dim_dual():
    link_arch = LinkArchDimDual()
    check_link_arch(link_arch, attempt_duration=(1, 1), notify_a=(1, 1), notify_b=(1, 1))


def test_sr():
    link_arch = LinkArchSr()
    check_link_arch(link_arch, attempt_duration=(2, 1), notify_a=(1, 1), notify_b=(2, 1))


def test_sim():
    link_arch = LinkArchSim()
    check_link_arch(link_arch, attempt_duration=(1, 1), notify_a=(1, 1), notify_b=(1, 1))
