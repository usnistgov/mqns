from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, overload, override

import numpy as np

if TYPE_CHECKING:
    from mqns.models.epr import MixedStateEntanglement, WernerStateEntanglement
    from mqns.models.qubit import Qubit


class ErrorModel(ABC):
    """
    ErrorModel models how a qubit or entanglement would receive noise.
    """

    def __init__(self, name: str):
        self.name = name
        """Name of this error model."""
        self._p_survival = 1.0
        self._last_rate = 0.0

    @property
    def p_survival(self) -> float:
        """Survival probability."""
        return self._p_survival

    @property
    def p_error(self) -> float:
        """Error probability."""
        return 1 - self._p_survival

    @overload
    def set(self, *, p_survival: float) -> Self:
        """
        Set survival probability.

        Args:
            p_survival: survival probability.
        """

    @overload
    def set(self, *, p_error: float) -> Self:
        """
        Set error probability.

        Args:
            p_error: error probability.
        """

    @overload
    def set(self, *, t: float, rate: float | None = None) -> Self:
        """
        Set time based decay.

        Args:
            t: duration stored in memory, in seconds.
            rate: decoherence rate in Hz; ``None`` to reuse last value.

        It's possible to use other time units, as long as ``t`` and ``rate`` are inverse of each other.
        """

    @overload
    def set(self, *, length: float, rate: float | None = None) -> Self:
        """
        Set length based decay.

        Args:
            length: distance traversed in channel, in km.
            rate: decoherence rate in ``km^-1``; ``None`` to reuse last value.

        It's possible to use other length units, as long as ``length`` and ``rate`` are inverse of each other.
        """

    def set(self, **kwargs) -> Self:
        return self._set(**kwargs)

    def _set(
        self,
        *,
        p_survival=1.0,
        p_error: float | None = None,
        t: float | None = None,
        length: float | None = None,
        rate: float | None = None,
    ) -> Self:
        if rate is None:
            rate = self._last_rate
        else:
            self._last_rate = rate

        if p_error is not None:
            p_survival = 1 - p_error
        elif t is not None:
            p_survival = np.exp(-rate * t)
        elif length is not None:
            p_survival = np.exp(-rate * length)

        assert 0 <= p_survival <= 1, "Survival/error probability must be between 0 and 1"
        if self._p_survival != p_survival:
            self._p_survival = p_survival
            self._prepare()

        return self

    def _prepare(self) -> None:
        """
        Invoked after error/survival probabilities are changed.
        Subclass may override to precompute values.
        """

    @abstractmethod
    def qubit(self, q: "Qubit") -> None:
        """
        Apply error model to a qubit.
        """

    @abstractmethod
    def werner(self, q: "WernerStateEntanglement") -> None:
        """
        Apply error model to a werner state entanglement.
        """

    @abstractmethod
    def mixed(self, q: "MixedStateEntanglement") -> None:
        """
        Apply error model to a mixed state entanglement.
        """


class PerfectErrorModel(ErrorModel):
    """
    Perfect error model: no error is applied.
    """

    def __init__(self, name="perfect"):
        super().__init__(name)

    @override
    def qubit(self, q) -> None:
        _ = q

    @override
    def werner(self, q) -> None:
        _ = q

    @override
    def mixed(self, q) -> None:
        _ = q
