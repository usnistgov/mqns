import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NotRequired, Protocol, TypedDict, Unpack, override

from mqns.entity.node import QNode
from mqns.models.delay import DelayModel
from mqns.models.epr import Entanglement, EntanglementInitKwargs, MixedStateEntanglement, WernerStateEntanglement
from mqns.models.error import DepolarErrorModel, ErrorModel, TimeDecayFunc, time_decay_nop
from mqns.models.error.input import ErrorModelInputBasic, parse_error
from mqns.simulator import Time

type MakeEprFunc = Callable[[EntanglementInitKwargs], Entanglement]


class QchannelParameters(Protocol):
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
    transfer_error: ErrorModel
    """
    Fiber transfer error model.
    This is only used if ``init_fidelity`` is omitted or negative.

    If the LinkArch subclass needs to apply transfer error at a different length,
    it will clone the instance and adjust the length while preserving the decoherence rate.
    """


class LinkArchParameters(TypedDict):
    ch: QchannelParameters
    """QuantumChannel to gather parameters from."""
    eta_s: float
    """Source efficiency between 0 and 1."""
    eta_d: float
    """Detector efficiency between 0 and 1."""
    reset_time: float
    """Inverse of source frequency in Hz."""
    tau_0: float
    """Local operation delay in seconds."""
    epr_type: type[Entanglement]
    """EPR type, either ``WernerStateEntanglement`` or ``MixedStateEntanglement``."""
    init_fidelity: NotRequired[float]
    """
    Initial fidelity value, defaults to ``-1``.

    If omitted or negative, a mini simulation is performed to determine the proper fidelity values,
    by applying error models to the memories, fibers, etc.

    If non-negative, every entanglement has a Werner state with the specified fidelity.
    """
    t0: NotRequired[Time]
    """
    Time reference for mini simulation; only the accuracy is used.
    This is required if ``init_fidelity`` is omitted or negative.
    """
    store_decays: NotRequired[tuple[TimeDecayFunc, TimeDecayFunc]]
    """
    Memory time-based decay functions at src and dst, defaults to perfect.
    This must accept the same time accuracy as ``t0``.
    This is only used if ``init_fidelity`` is omitted or negative.

    Current limitation: if a qchannel is activated in two paths with opposite directions,
    and the two memories have different error models, the calculations would be incorrect.
    """
    bsa_error: NotRequired[ErrorModelInputBasic]
    """
    Bell-state analyzer or absorptive memory capture error model, defaults to perfect.
    This is only used if ``init_fidelity`` is omitted or negative.
    """


class LinkArch(Protocol):
    """
    Link architecture models the elementary entanglement generation protocol.

    Together with quantum channel and node hardware parameters, it supplies information to
    the skip-ahead sampling implementation in ``LinkLayer`` application.
    """

    name: str
    """Link architecture name."""

    success_prob: float
    """
    Success probability of a single attempt.

    This is available after `set()`.
    """

    def set(self, **kwargs: Unpack[LinkArchParameters]) -> None:
        """
        Save parameters about quantum channel and node hardware.
        """

    def delays(self, k: int) -> tuple[float, float, float]:
        """
        Compute protocol delays for k-th attempt.
        This is available after `set()`.

        Args:
            k: number of attempts, minimum is 1.

        Returns:
            Each value is a duration in seconds.

            * [0]: EPR creation time, since RESERVE_QUBIT_OK arrives at primary node.
            * [1]: notification time to primary node, since EPR creation.
            * [2]: notification time to secondary node, since EPR creation.
        """
        ...

    def make_epr(self, k: int, now: Time, *, key: str | None, src: QNode, dst: QNode) -> tuple[Entanglement, Time, Time]:
        """
        Create an elementary entanglement for k-th attempt.
        This is available after `set()`.

        Args:
            k: number of attempts, minimum is 1.
            now: current time point, which is when RESERVE_QUBIT_OK arrives at primary node.
            key: LinkLayer reservation key.
            src: primary node.
            dst: secondary node.

        Returns:
            Each value is a duration in seconds.

            * [0]: EPR object with fidelity assigned.
                   Its ``fidelity_time`` may be different from "EPR creation time" returned by ``delays()``.
            * [1]: notification time point to primary node.
            * [2]: notification time point to secondary node.
        """
        ...


class LinkArchBase(ABC, LinkArch):
    def __init__(self, name: str):
        self.name = name
        self.success_prob = 0.0
        self.attempt_interval = 0.0
        self.d_notify_a = 0.0
        self.d_notify_b = 0.0

    @override
    def set(self, **kwargs: Unpack[LinkArchParameters]) -> None:
        ch = kwargs["ch"]
        tau_l = ch.delay.calculate()
        for _ in range(16):
            assert ch.delay.calculate() == tau_l, "QuantumChannel.delay must be constant"

        self.success_prob = self._compute_success_prob(
            length=ch.length,
            alpha=ch.alpha,
            eta_s=kwargs["eta_s"],
            eta_d=kwargs["eta_d"],
        )

        self.attempt_interval, self.d_notify_a, self.d_notify_b = self._compute_delays(
            reset_time=kwargs["reset_time"],
            tau_l=tau_l,
            tau_0=kwargs["tau_0"],
        )

        epr_type = kwargs["epr_type"]
        init_fidelity = kwargs.get("init_fidelity", -1)

        def _make_epr_with_init_fidelity(a: EntanglementInitKwargs) -> Entanglement:
            epr = epr_type(**a)
            epr.fidelity = init_fidelity
            return epr

        if init_fidelity < 0:
            self._make_epr: MakeEprFunc = self._prepare_make_epr(kwargs, ch, tau_l)
        else:
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
        Override ``delays()`` method for unusual situations.
        """

    @override
    def delays(self, k: int) -> tuple[float, float, float]:
        return (k - 1) * self.attempt_interval, self.d_notify_a, self.d_notify_b

    def _prepare_make_epr(self, d: LinkArchParameters, ch: QchannelParameters, tau_l: float) -> MakeEprFunc:
        accuracy = d.get("t0", Time.SENTINEL).accuracy
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
            bsa=parse_error(d.get("bsa_error"), DepolarErrorModel),
        )
        assert type(epr) is epr_type

        # The final state could reflect any time point between EPR creation time and the earlier heralding time.
        t_diff = epr.fidelity_time - t0
        assert Time(0, accuracy=accuracy) <= t_diff <= Time.from_sec(min(self.d_notify_a, self.d_notify_b), accuracy=accuracy)

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

    @override
    def make_epr(self, k: int, now: Time, *, key: str | None, src: QNode, dst: QNode) -> tuple[Entanglement, Time, Time]:
        d_epr_creation, d_notify_a, d_notify_b = self.delays(k)
        t_epr_creation = now + d_epr_creation
        t_notify_a = now + (d_epr_creation + d_notify_a)
        t_notify_b = now + (d_epr_creation + d_notify_b)

        mem_a, mem_b = src.memory, dst.memory
        epr = self._make_epr(
            EntanglementInitKwargs(
                decohere_time=t_epr_creation + min(mem_a.decoherence_delay, mem_b.decoherence_delay),
                fidelity_time=t_epr_creation,
                src=src,
                dst=dst,
                store_decays=(mem_a.time_decay, mem_b.time_decay),
            )
        )
        epr.key = key

        return epr, t_notify_a, t_notify_b


class LinkArchAlways(LinkArch):
    """
    Link architecture wrapper that always succeeds, primarily for unit testing.
    """

    def __init__(self, inner: LinkArch):
        self.name = f"{inner.name}-always"
        self.inner = inner

    @override
    def set(self, **kwargs: Unpack[LinkArchParameters]) -> None:
        self.inner.set(**kwargs)
        self.success_prob = 1.0

    @override
    def delays(self, k: int) -> tuple[float, float, float]:
        assert k == 1
        return self.inner.delays(k)

    @override
    def make_epr(self, k: int, *args, **kwargs):
        assert k == 1
        return self.inner.make_epr(k, *args, **kwargs)
