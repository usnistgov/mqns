from collections.abc import Iterable
from typing import Self, override

from mqns.models.error.error import ErrorModel


class ChainErrorModel(ErrorModel):
    """
    Chain that composites multiple error models.

    The ``set`` method sets the same parameters onto each enclosed error model.
    To set different parameters, call ``set`` on each enclosed error model directly.

    ``p_survival`` and ``p_error`` properties are disabled because the chain can consist of
    a mix of different physical noise processes that cannot be described by a scalar probability.
    """

    def __init__(self, errors: Iterable[ErrorModel]):
        """
        Constructor:

        Args:
            errors: list of error models applied in sequential order.
        """
        self.errors = list(errors)
        super().__init__("CHAIN(" + ",".join(m.name for m in self.errors) + ")")

    @property
    @override
    def p_survival(self) -> float:
        raise TypeError("cannot retrieve survival probability in ChainErrorModel")

    @property
    @override
    def p_error(self) -> float:
        raise TypeError("cannot retrieve error probability in ChainErrorModel")

    @override
    def _set(self, **kwargs) -> Self:
        for m in self.errors:
            m.set(**kwargs)
        return self

    @override
    def qubit(self, q) -> None:
        for m in self.errors:
            m.qubit(q)

    @override
    def werner(self, q) -> None:
        for m in self.errors:
            m.werner(q)

    @override
    def mixed(self, q) -> None:
        for m in self.errors:
            m.mixed(q)
