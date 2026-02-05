from typing import override

from mqns.entity.base_channel import calc_transmission_prob
from mqns.entity.qchannel.link_arch import LinkArchBase
from mqns.models.epr import Entanglement
from mqns.models.error import ErrorModel, TimeDecayFunc
from mqns.simulator import Time


class LinkArchSr(LinkArchBase):
    """
    Sender-Receiver link architecture with dual-rail polarization encoding.

    The receiver is modeled as direct absorption: when a photon hits its detector, the memory captures its state.
    """

    def __init__(self, name="SR"):
        super().__init__(name)

    @override
    def _compute_success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        # The success probability consists of:
        # - eta_s: the source at B emits a photon.
        # - p_l_sr: the photon propagates through the fiber without loss.
        # - eta_d: the detector at A detects the photon.
        p_l_sr = calc_transmission_prob(length, alpha)
        eta_sr = eta_s * eta_d * p_l_sr
        return eta_sr

    @override
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        # Reservation and setup were completed by LinkLayer.
        # In each attempt:
        # 1. Qubits are generated at +0
        # 2. B emits a photon at +τ0, which arrives at +τl+τ0
        # 3. A absorbs the photon at +τl+τ0
        # 4. A sends heralding result to B, which arrives at +2τl+τ0
        # If the heralding result indicates failure, the next attempt can immediately start.
        # The attempt interval is lower bounded by reset_time for one memory excitation.
        attempt_duration = 2 * tau_l + tau_0
        attempt_interval = max(attempt_duration, reset_time)
        return attempt_interval, tau_l + tau_0, attempt_duration

    @override
    def _simulate_errors(
        self,
        *,
        epr_type: type[Entanglement],
        tc: Time,
        store_src: TimeDecayFunc,
        store_dst: TimeDecayFunc,
        tau_l: Time,
        tau_0: Time,
        transfer_full: ErrorModel,
        bsa: ErrorModel,
        **_,
    ) -> Entanglement:
        # B generates memory-photon entanglement and sends the photon to A.
        epr = epr_type(fidelity_time=tc, store_decays=(None, store_dst))

        # The photon travels to A.
        tc += tau_0 + tau_l
        epr.apply_error(transfer_full)

        # B's qubit decays while the photon is in flight.
        epr.apply_store_decays(tc)

        # A captures the photon and it starts to decay.
        epr.apply_error(bsa)
        epr.store_decays = (store_src, store_dst)

        return epr
