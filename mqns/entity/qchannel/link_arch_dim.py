from typing import override

from mqns.entity.base_channel import calc_transmission_prob
from mqns.entity.qchannel.link_arch import LinkArchBase
from mqns.models.epr import Entanglement
from mqns.models.error import ErrorModel, TimeDecayFunc
from mqns.simulator import Time


class LinkArchDimBk(LinkArchBase):
    """
    Detection-in-Midpoint link architecture with single-rail encoding using Barrett-Kok protocol.
    """

    def __init__(self, name="DIM-BK"):
        super().__init__(name)

    @override
    def _compute_success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        # Barrett-Kok uses single-rail encoding where the presence/absence of a photon indicates quantum state.
        # For a successful attempt, exactly one photon should arrive at the Bell-state analyzer M in each round.
        #
        # eta_sb is the probability of a photon triggering a detector, which consists of:
        # - eta_s: the source at A or B emits a photon.
        # - p_l_sb: the photon propagates through the fiber without loss.
        # - eta_d: the detector at M detects the photon.
        #
        # p_bsa, set to 50%, is the maximum theoretical coincidence probability for distinguishing two of
        # the four Bell states at a standard linear optics Bell-state analyzer.
        p_bsa = 0.5
        p_l_sb = calc_transmission_prob(length / 2, alpha)
        eta_sb = eta_s * eta_d * p_l_sb
        return p_bsa * eta_sb**2

    @override
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        # Reservation and setup were completed by LinkLayer.
        # In each attempt:
        # 1. Qubits are generated at +0
        # 2. Both A and B emit photons at +τ0, which arrive at the Bell-state analyzer M at +(1/2)τl+τ0
        # 3. M sends heralding results to both A and B, which arrive at +τl+τ0
        # 4. Both A and B flip qubit gates locally
        # 5. Both A and B emit photons at +τl+2τ0, which arrive at M at +(1 1/2)τl+2τ0
        # 6. M sends heralding results to both A and B, which arrive at +2τl+2τ0
        # If either heralding result indicates failure, the next attempt can immediately start.
        # The attempt interval is lower bounded by twice of reset_time for two memory excitations.
        attempt_duration = 2 * (tau_l + tau_0)
        attempt_interval = max(attempt_duration, 2 * reset_time)
        return attempt_interval, attempt_duration, attempt_duration

    @override
    def _simulate_errors(
        self,
        *,
        epr_type: type[Entanglement],
        tc: Time,
        store_src: TimeDecayFunc,
        store_dst: TimeDecayFunc,
        tau_l2: Time,
        tau_0: Time,
        transfer_half: ErrorModel,
        bsa: ErrorModel,
        **_,
    ) -> Entanglement:
        # A and B excite their memories and emit round-1 photon(s).
        e0 = epr_type(fidelity_time=tc, store_decays=(store_src, None))
        e1 = epr_type(fidelity_time=tc, store_decays=(None, store_dst))

        # Round-1 photon(s) arrive at the Bell-state analyzer.
        tc += tau_0 + tau_l2
        # Although physically there is only one photon, both potential paths are noisy,
        # so that both EPRs are subject to transfer errors.
        e0.apply_error(transfer_half)
        e1.apply_error(transfer_half)
        e2 = Entanglement.swap(e0, e1, now=tc)
        del e0, e1
        assert e2 is not None
        e2.apply_error(bsa)

        # In round-2, physically each of A and B generates a memory-photo EPR that collects noise
        # and then swaps with e2, but this is equivalent to applying noise onto e2 directly.
        e2.apply_error(transfer_half)
        e2.apply_error(transfer_half)
        e2.apply_error(bsa)

        return e2


class LinkArchDimBkSeq(LinkArchDimBk):
    """
    Detection-in-Midpoint link architecture with single-rail encoding using Barrett-Kok protocol,
    timing adjusted as per negotiation logic implemented by SeQUeNCe simulator.
    """

    def __init__(self, name="DIM-BK-SeQUeNCe"):
        super().__init__(name)

    @override
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        # According to SeQUeNCe logic:
        # 1. A and B perform the first round negotiation, which completes at +2τl
        # 2. Qubits are generated at +2τl
        # 3. Both A and B emit photons at +2τl+τ0, which arrive at M at +(2 1/2)τl+τ0
        # 4. M sends heralding results to both A and B, which arrive at +3τl+τ0
        # 5. If the heralding result indicates failure, the current attempt is aborted
        # 6. A and B perform the second round negotiation, which completes at +5τl+τ0
        # 7. Both A and B flip qubit gates locally
        # 8. Both A and B emit photons at +5τl+2τ0, which arrive at M at +(5 1/2)τl+2τ0
        # 9. M sends heralding results to both A and B, which arrive at +6τl+2τ0
        # In summary, success occurs at +6τl+2τ0, failure occurs at either +3τl+τ0 or +6τl+2τ0.
        #
        # This model does not differentiate the two failure possibilities.
        # The attempt duration is set to 5τl+2τ0, in between two possible failed attempt durations.
        # It is also lower bounded by twice of reset_time for two memory excitations.
        #
        # The first round negotiation in the initial attempt, which takes 2τl, overlaps the reservation logic
        # in LinkLayer, so that the initial attempt is 2τl shorter.
        # For simplicity, this shortening is applied on the final attempt instead of the initial attempt.
        # Thus, the final attempt succeeds at +4τl+2τ0 as calculated in d_notify.
        attempt_duration = 5 * tau_l + 2 * tau_0
        attempt_interval = max(attempt_duration, 2 * reset_time)
        d_notify = 4 * tau_l + 2 * tau_0
        return attempt_interval, d_notify, d_notify


class LinkArchDimDual(LinkArchBase):
    """
    Detection-in-Midpoint link architecture with dual-rail polarization encoding.
    """

    def __init__(self, name="DIM-dual"):
        super().__init__(name)

    @override
    def _compute_success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        # For a successful attempt, one photon from each of A and B should arrive at the Bell-state analyzer M.
        #
        # eta_sb is the probability of a photon triggering a detector, which consists of:
        # - eta_s: the source at A or B emits a photon.
        # - p_l_sb: the photon propagates through the fiber without loss.
        # - eta_d: the detector at M detects the photon.
        #
        # p_bsa, set to 50%, is the maximum theoretical coincidence probability for distinguishing two of
        # the four Bell states at a standard linear optics Bell-state analyzer.
        p_bsa = 0.5
        p_l_sb = calc_transmission_prob(length / 2, alpha)
        eta_sb = eta_s * eta_d * p_l_sb
        return p_bsa * eta_sb**2

    @override
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        # Reservation and setup were completed by LinkLayer.
        # In each attempt:
        # 1. Qubits are generated at +0
        # 2. Both A and B emit photons at +τ0, which arrive at the Bell-state analyzer M at +(1/2)τl+τ0
        # 3. M sends heralding results to both A and B, which arrive at +τl+τ0
        # If the heralding result indicates failure, the next attempt can immediately start.
        # The attempt interval is lower bounded by reset_time for one memory excitation.
        attempt_duration = tau_l + tau_0
        attempt_interval = max(attempt_duration, reset_time)
        return attempt_interval, attempt_duration, attempt_duration

    @override
    def _simulate_errors(
        self,
        *,
        epr_type: type[Entanglement],
        tc: Time,
        store_src: TimeDecayFunc,
        store_dst: TimeDecayFunc,
        tau_l2: Time,
        tau_0: Time,
        transfer_half: ErrorModel,
        bsa: ErrorModel,
        **_,
    ) -> Entanglement:
        # A and B generate memory-photon entanglements.
        e0 = epr_type(fidelity_time=tc, store_decays=(store_src, None))
        e1 = epr_type(fidelity_time=tc, store_decays=(None, store_dst))

        # Photons arrive at the Bell-state analyzer.
        tc += tau_0 + tau_l2
        e0.apply_error(transfer_half)
        e1.apply_error(transfer_half)
        e2 = Entanglement.swap(e0, e1, now=tc)
        del e0, e1
        assert e2 is not None
        e2.apply_error(bsa)

        return e2
