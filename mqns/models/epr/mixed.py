from collections.abc import Iterable
from typing import Literal, Unpack, cast, final, overload, override

import numpy as np

from mqns.models.epr.entanglement import Entanglement, EntanglementInitKwargs
from mqns.models.qubit.state import (
    ATOL,
    BELL_RHO_PHI_N,
    BELL_RHO_PHI_P,
    BELL_RHO_PSI_N,
    BELL_RHO_PSI_P,
    QubitRho,
    check_qubit_rho,
)
from mqns.utils import rng

ProbabilityVector = np.ndarray[tuple[Literal[4]], np.dtype[np.float64]]


@final
class MixedStateEntanglement(Entanglement["MixedStateEntanglement"]):
    """A pair of entangled qubits in Bell-Diagonal State with a hidden-variable."""

    @overload
    def __init__(self, *, fidelity=1.0, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Construct with fidelity.

        This creates a Werner state where the error probabilities are distributed equally among the three bad states.
        """
        pass

    @overload
    def __init__(self, *, i: float, z: float, x: float, y: float, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Construct with four probability values.

        Args:
            i: Probability of desired state, i.e. fidelity.
            z: Probability of Z-flip.
            x: Probability of X-flip.
            y: Probability of Y-flip.
        """
        pass

    def __init__(self, *, fidelity: float | None = None, i=1.0, z=0.0, x=0.0, y=0.0, **kwargs: Unpack[EntanglementInitKwargs]):
        super().__init__(**kwargs)
        if fidelity is None:
            self.set_prob(i, z, x, y)
        else:
            self.fidelity = fidelity

    @property
    @override
    def fidelity(self) -> float:
        return self._prob[0].item()

    @fidelity.setter
    @override
    def fidelity(self, value: float):
        """Reset fidelity, turning into a Werner state."""
        bcd = (1 - value) / 3
        self.set_prob(value, bcd, bcd, bcd)

    @property
    def i(self) -> float:
        """Probability of desired state."""
        return self._prob[0]

    @property
    def z(self) -> float:
        """Probability of Z-flip."""
        return self._prob[1]

    @property
    def x(self) -> float:
        """Probability of X-flip."""
        return self._prob[2]

    @property
    def y(self) -> float:
        """Probability of Y-flip."""
        return self._prob[3]

    def set_prob(self, i: float, z: float, x: float, y: float) -> None:
        """Update probability values."""
        self._set_prob(np.array((i, z, x, y), dtype=np.float64))

    def _set_prob(self, prob: ProbabilityVector) -> None:
        total = np.sum(prob)
        if total <= ATOL:  # avoid divide-by-zero
            self._prob = cast(ProbabilityVector, np.zeros(4, dtype=np.float64))
        else:
            self._prob = cast(ProbabilityVector, prob / total)

    @staticmethod
    @override
    def _make_swapped(epr0: "MixedStateEntanglement", epr1: "MixedStateEntanglement", **kwargs: Unpack[EntanglementInitKwargs]):
        i0, z0, x0, y0 = epr0._prob
        m_swap = np.array(
            [
                [i0, z0, x0, y0],  # i row: (i0*i1, z0*z1, x0*x1, y0*y1)
                [z0, i0, y0, x0],  # z row: (i0*z1, z0*i1, x0*y1, y0*x1)
                [x0, y0, i0, z0],  # x row: (i0*x1, z0*y1, x0*i1, y0*z1)
                [y0, x0, z0, i0],  # y row: (i0*y1, z0*x1, x0*z1, y0*i1)
            ],
            dtype=np.float64,
        )
        i2, z2, x2, y2 = m_swap @ epr1._prob
        return MixedStateEntanglement(i=i2, z=z2, x=x2, y=y2, **kwargs)

    @override
    def _do_purify(self, epr1: "MixedStateEntanglement") -> bool:
        """
        Perform distillation using BBPSSW protocol.
        """
        i0, z0, x0, y0 = self._prob
        i1, z1, x1, y1 = epr1._prob
        p_succ = (i0 + y0) * (i1 + y1) + (z0 + x0) * (x1 + z1)
        if p_succ <= ATOL or rng.random() > p_succ:
            return False

        self.set_prob(
            i0 * i1 + y0 * y1,
            z0 * z1 + x0 * x1,
            z0 * x1 + x0 * z1,
            i0 * y1 + y0 * i1,
        )
        return True

    def dephase(self, t: float, rate: float):
        """
        Inject dephasing noise.

        Args:
            t: time in seconds, distance in km, etc.
            rate: dephasing rate, unit is inverse of ``t``.
        """
        i, z, x, y = self._prob
        multiplier = np.exp(-rate * t)
        self.set_prob(
            0.5 + (i - 0.5) * multiplier,
            0.5 + (z - 0.5) * multiplier,
            x,
            y,
        )

    def depolarize(self, t: float, rate: float):
        """
        Inject depolarizing noise.

        Args:
            t: time in seconds, distance in km, etc.
            rate: depolarizing rate, unit is inverse of ``t``.
        """
        multiplier = np.exp(-rate * t)
        self._set_prob(0.25 + (self._prob - 0.25) * multiplier)

    @override
    def store_error_model(self, t: float = 0, decoherence_rate: float = 0, **kwargs):
        """
        Apply an error model for storing this entangled pair in a quantum memory.

        Args:
            t: duration since last update in seconds.
            decoherence_rate: memory decoherence rate in Hz.
        """
        _ = kwargs
        self.depolarize(t, decoherence_rate)

    @override
    def transfer_error_model(self, length: float = 0, decoherence_rate: float = 0, **kwargs):
        """
        Apply an error model for transmitting this entanglement.

        Args:
            length: channel length in km.
            decoherence_rate: channel decoherence rate in km^-1.
        """
        _ = kwargs
        self.depolarize(length, decoherence_rate)

    @override
    def _to_qubits_rho(self) -> QubitRho:
        i, z, x, y = self._prob
        return check_qubit_rho(i * BELL_RHO_PHI_P + z * BELL_RHO_PHI_N + x * BELL_RHO_PSI_P + y * BELL_RHO_PSI_N, n=2)

    @override
    def _describe_fidelity(self) -> Iterable[str]:
        i, z, x, y = self._prob
        yield f"i={i:.4f}"
        yield f"z={z:.4f}"
        yield f"x={x:.4f}"
        yield f"y={y:.4f}"
