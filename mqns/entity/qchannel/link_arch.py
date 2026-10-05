import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Final, NotRequired, Protocol, TypedDict, Unpack

from mqns.entity.node import QNode
from mqns.models.core.bell_diagonal import make_bell_diagonal_probv
from mqns.models.delay import ConstantDelayModel, DelayModel
from mqns.models.epr import Entanglement, EntanglementInitKwargs, MixedStateEntanglement, WernerStateEntanglement
from mqns.models.error import ErrorModel, TimeDecayFunc, time_decay_nop
from mqns.simulator import Time

type _MakeEprFunc = Callable[[EntanglementInitKwargs], Entanglement]


class ChannelParameters(Protocol):
    """QuantumChannel parameters related to LinkArch."""

    length: float
    """Fiber length in kilometers."""
    alpha: float
    """Fiber attenuation loss in dB/km."""
    delay: DelayModel
    """
    Fiber propagation delay in seconds, also used as one-way classical message delay.
    This must reflect a constant delay.
    """
    eta_s: float
    """Source efficiency between 0 and 1."""
    eta_d: float
    """Detector efficiency between 0 and 1."""
    init_fidelity: float | Sequence[float] | None
    """Initial fidelity value for entanglements delivered by this channel."""
    transfer_error: ErrorModel
    """
    Fiber transfer error model.
    This is only used if ``init_fidelity`` is omitted or negative.

    If the ``LinkArch`` subclass needs to apply transfer error at a different length,
    it will clone the instance and adjust the length while preserving the decoherence rate.
    """
    bsa_error: ErrorModel
    """
    Bell-state analyzer or absorptive memory capture error model.
    This is only used if ``init_fidelity`` is omitted or negative.
    """


class LinkArchParameters(TypedDict):
    time_accuracy: int
    """Time accuracy."""
    ch: ChannelParameters
    """QuantumChannel to gather parameters from."""
    reset_time: float
    """Inverse of source frequency in Hz."""
    tau_0: float
    """Local operation delay in seconds."""
    epr_type: type[Entanglement]
    """EPR type, either ``WernerStateEntanglement`` or ``MixedStateEntanglement``."""
    store_decays: NotRequired[tuple[TimeDecayFunc, TimeDecayFunc]]
    """
    Memory time-based decay functions at src and dst, defaults to perfect.
    This must accept the same time accuracy as ``t0``.
    This is only used if ``init_fidelity`` is omitted.

    Current limitation: if a qchannel is activated in two paths with opposite directions,
    and the two memories have different error models, the calculations would be incorrect.
    """


class LinkArch(Protocol):
    """
    Link architecture models the elementary entanglement generation protocol.

    Together with quantum channel and node hardware parameters, it supplies information to
    the skip-ahead sampling implementation in ``LinkLayer`` application.

    All properties and methods, other than ``name``, are only available after ``set()``.
    """

    @property
    def name(self) -> str:
        """Link architecture name."""
        ...

    def set(self, **kwargs: Unpack[LinkArchParameters]) -> None:
        """Save parameters about quantum channel and node hardware."""

    @property
    def success_prob(self) -> float:
        """Success probability of a single attempt."""
        ...

    @property
    def attempt_interval(self) -> Time:
        """
        How often can the LinkLayer make a new attempt.

        The k-th attempt shall begin at ``(k-1) * attempt_interval``.
        """
        ...

    @property
    def d_notify_pri(self) -> Time:
        """How soon is the primary node notified for entanglement since a successful attempt began."""
        ...

    @property
    def d_notify_2nd(self) -> Time:
        """How soon is the secondary node notified for entanglement since a successful attempt began."""
        ...

    def make_epr(self, t_epr_creation: Time, src: QNode, dst: QNode, *, key: str | None) -> Entanglement:
        """
        Create an elementary entanglement.

        Args:
            t_epr_creation: EPR creation time, i.e. the start time of the successful attempt.
            src: Primary node.
            dst: Secondary node.
            key: Memory qubit reservation key.

        Returns:
            EPR object with fidelity assigned.
            Its ``fidelity_time`` may be different from EPR creation time.
        """
        ...


def assert_is_link_arch[T: LinkArch](typ: type[T]) -> type[T]:
    """Ensure a type properly implements ``LinkArch`` protocol."""
    return typ


class LinkArchBase(ABC):
    name: Final[str]
    success_prob = 0.0
    attempt_interval = Time.SENTINEL
    d_notify_pri = Time.SENTINEL
    d_notify_2nd = Time.SENTINEL
    _make_epr: _MakeEprFunc

    def __init__(self, name: str):
        self.name = name

    def set(self, **kwargs: Unpack[LinkArchParameters]) -> None:
        accuracy = kwargs["time_accuracy"]
        ch = kwargs["ch"]
        tau_l = ConstantDelayModel.extract(ch.delay)

        self.success_prob = self._compute_success_prob(
            length=ch.length,
            alpha=ch.alpha,
            eta_s=ch.eta_s,
            eta_d=ch.eta_d,
        )

        attempt_interval, d_notify_pri, d_notify_2nd = self._compute_delays(
            reset_time=kwargs["reset_time"],
            tau_l=tau_l,
            tau_0=kwargs["tau_0"],
        )
        self.attempt_interval = Time.from_sec(attempt_interval, accuracy=accuracy)
        self.d_notify_pri = Time.from_sec(d_notify_pri, accuracy=accuracy)
        self.d_notify_2nd = Time.from_sec(d_notify_2nd, accuracy=accuracy)

        if (init_fidelity := ch.init_fidelity) is None:
            self._make_epr = self._prepare_make_epr(kwargs, ch, tau_l)
        elif isinstance(init_fidelity, Sequence):
            assert kwargs["epr_type"] is MixedStateEntanglement
            assert len(init_fidelity) == 4
            probv = make_bell_diagonal_probv(*init_fidelity)

            def _make_epr_with_probv(a: EntanglementInitKwargs) -> Entanglement:
                epr = MixedStateEntanglement(**a)
                epr.set_probv(probv, normalize=False)
                return epr

            self._make_epr = _make_epr_with_probv
        else:
            epr_type = kwargs["epr_type"]
            assert 0 <= init_fidelity <= 1

            def _make_epr_with_init_fidelity(a: EntanglementInitKwargs) -> Entanglement:
                epr = epr_type(**a)
                epr.fidelity = init_fidelity
                return epr

            self._make_epr = _make_epr_with_init_fidelity

    @abstractmethod
    def _compute_success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        """
        Compute success probability of a single attempt.
        Subclass implementation may precompute or save other parameters if necessary.
        """

    @abstractmethod
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        """
        Compute attempt interval and notification delays, for protocol delay computation.
        Subclass implementation may precompute or save other parameters if necessary.
        """

    def _prepare_make_epr(self, d: LinkArchParameters, ch: ChannelParameters, tau_l: float) -> _MakeEprFunc:
        accuracy = d["time_accuracy"]
        t0 = Time.from_sec(1, accuracy=accuracy)
        epr_type = d["epr_type"]
        se0, se1 = d.get("store_decays", (time_decay_nop, time_decay_nop))

        # Perform a mini simulation to calculate the state of heralded entanglement.
        epr = self._simulate_errors(
            epr_type=epr_type,
            tc=t0,
            reset_time=Time.from_sec(d["reset_time"], accuracy=accuracy),
            tau_l=Time.from_sec(tau_l, accuracy=accuracy),
            tau_l2=Time.from_sec(tau_l / 2, accuracy=accuracy),
            tau_0=Time.from_sec(d["tau_0"], accuracy=accuracy),
            store_src=se0,
            store_dst=se1,
            transfer_full=ch.transfer_error,
            transfer_half=copy.deepcopy(ch.transfer_error).set(length=ch.length / 2),
            bsa=ch.bsa_error,
        )
        assert type(epr) is epr_type

        # The final state could reflect any time point between EPR creation time and the earlier heralding time.
        t_diff = epr.fidelity_time - t0
        assert Time(0, accuracy=accuracy) <= t_diff <= min(self.d_notify_pri, self.d_notify_2nd)

        # Capture the final state.
        update = {}
        if type(epr) is WernerStateEntanglement:
            update["w"] = epr.w
        elif type(epr) is MixedStateEntanglement:
            update["probv"] = epr.probv
        else:
            raise TypeError("unsupported EPR type")

        def _make_epr_adjusted(a: EntanglementInitKwargs) -> Entanglement:
            # Copy final state and adjust fidelity_time.
            if "fidelity_time" in a:
                a["fidelity_time"] += t_diff
            a.update(update)
            return epr_type(**a)

        return _make_epr_adjusted

    @abstractmethod
    def _simulate_errors(
        self,
        *,
        epr_type: type[Entanglement],
        tc: Time,
        reset_time: Time,
        store_src: TimeDecayFunc,
        store_dst: TimeDecayFunc,
        tau_l: Time,
        tau_l2: Time,
        tau_0: Time,
        transfer_full: ErrorModel,
        transfer_half: ErrorModel,
        bsa: ErrorModel,
    ) -> Entanglement:
        """
        Perform a mini simulation to establish elementary entanglement between two nodes,
        applying error models along the way.

        Args:
            epr_type: Entanglement type.
            tc: Reference time point corresponding to ``t_epr_creation``.
            reset_time: Inverse of source frequency.
            store_src: Memory time-based decay function at primary node.
            store_dst: Memory time-based decay function at secondary node.
            tau_l: Fiber propagation delay for full length.
            tau_l2: Fiber propagation delay for half length.
            tau_0: Local propagation delay.
            transfer_full: Transfer error model for full length.
            transfer_half: Transfer error model for half length.
            bsa: Bell-state analyzer error model.

        Returns: the final entanglement.
        """

    def make_epr(self, t_epr_creation: Time, src: QNode, dst: QNode, *, key: str | None) -> Entanglement:
        mem_a, mem_b = src.memory, dst.memory
        return self._make_epr(
            EntanglementInitKwargs(
                decohere_time=t_epr_creation + min(mem_a.t_cohere, mem_b.t_cohere),
                fidelity_time=t_epr_creation,
                src=src,
                dst=dst,
                mem_keys=(key, key),
                store_decays=(mem_a.time_decay, mem_b.time_decay),
            )
        )


@assert_is_link_arch
class LinkArchAlways:
    """
    Link architecture wrapper that always succeeds, primarily for unit testing.
    """

    def __init__(self, inner: LinkArch):
        self.name = f"{inner.name}-always"
        self.inner = inner
        self.set = inner.set
        self.make_epr = inner.make_epr

    @property
    def success_prob(self):
        return 1.0

    @property
    def attempt_interval(self):
        return self.inner.attempt_interval

    @property
    def d_notify_pri(self):
        return self.inner.d_notify_pri

    @property
    def d_notify_2nd(self):
        return self.inner.d_notify_2nd
