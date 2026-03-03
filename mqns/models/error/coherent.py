from typing import override

import numpy as np

from mqns.models.core.operator import OPERATOR_RY
from mqns.models.error.error import ErrorModel
from mqns.utils import rng


class CoherentErrorModel(ErrorModel):
    """
    Coherent error model: distance-dependent unitary rotation.

    Note: cannot construct this type with ``parse_error`` utility.
    """

    def __init__(self, name="coherent", *, length=0.0, standard_lkm=50.0):
        super().__init__(name)
        self.standard_lkm = standard_lkm
        """Characteristic distance: the distance at which the maximum possible rotation angle reaches 45°."""
        self.set(length=length)

    @override
    def _set(self, **kwargs):
        length = kwargs.get("length")
        if length is None:
            raise TypeError("CoherentErrorModel must be set with length")
        self.length = float(length)
        """Fiber length in km."""
        self._prepare()
        return self

    @override
    def _prepare(self) -> None:
        self._max_theta = (self.length / self.standard_lkm) * (np.pi / 4)

    @override
    def qubit(self, q) -> None:
        theta = rng.uniform(0, self._max_theta)
        op = OPERATOR_RY(theta * 2)  # theta is physical radians, but RY uses Bloch Sphere radians
        q.state.operate(op)

    @override
    def werner(self, q) -> None:
        _ = q
        raise NotImplementedError()

    @override
    def mixed(self, q) -> None:
        _ = q
        raise NotImplementedError()
