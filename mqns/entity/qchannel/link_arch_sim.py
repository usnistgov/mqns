from typing import override

from mqns.entity.base_channel import calc_transmission_prob
from mqns.entity.qchannel.link_arch import LinkArchBase
from mqns.models.epr import Entanglement
from mqns.models.error import ErrorModel, TimeDecayFunc
from mqns.simulator import Time


class LinkArchSim(LinkArchBase):
    """
    Source-in-Midpoint link architecture with dual-rail polarization encoding.

    The receiver is modeled as direct absorption: when a photon hits its detector, the memory captures its state.
    """

    def __init__(self, name="SIM"):
        super().__init__(name)

    @override
    def _compute_success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        # For a successful attempt, A and B must each receive a photon from the pair.
        #
        # eta_d*p_l_sb is the probability of a photon reaching either A or B, which consists of:
        # - p_l_sb: the photon propagates through the fiber without loss.
        # - eta_d: the detector at A or B detects the photon.
        #
        # The overall success probability has `**2` because it requires both photons.
        _ = eta_s
        p_l_sb = calc_transmission_prob(length / 2, alpha)
        eta_rr = (eta_d * p_l_sb) ** 2
        return eta_rr

    @override
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        # Reservation and setup were completed by LinkLayer.
        # In each attempt:
        # 1. Entangled photon pairs are continuously emitted by the entangled photon source
        # 2. Both A and B prepare their local qubits at +0, which completes at +τ0
        # 3. A absorbs a photon at +τ0; B absorbs the paired photon at +τ0
        # 4. A sends heralding result to B, which arrives at +τl+τ0; B does the same in opposite direction
        # If either heralding result indicates failure, the next attempt can immediately start.
        # The attempt interval is lower bounded by reset_time for one local qubit preparation.
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
        # Entangled photon source emits the usable pair.
        tc -= tau_l2 - tau_0
        epr = epr_type(fidelity_time=tc, store_decays=(None, None))

        # The photons travel to A and B.
        tc += tau_l2
        epr.apply_error(transfer_half)
        epr.apply_error(transfer_half)

        # While the photons are in flight, they are not subject to memory decay,
        # so that apply_store_decays() would just update fidelity_time.
        epr.apply_store_decays(tc)

        # A and B each captures a photon and it starts to decay.
        epr.apply_error(bsa)
        epr.apply_error(bsa)
        epr.store_decays = (store_src, store_dst)

        return epr
