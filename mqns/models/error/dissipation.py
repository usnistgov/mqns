from typing import override

from mqns.models.core.state import QUBIT_RHO_0
from mqns.models.error.error import ErrorModel
from mqns.utils import rng


class DissipationErrorModel(ErrorModel):
    """
    Dissipation error model: qubit becomes ``|0>`` with ``p_error`` probability.
    """

    def __init__(self, name="dissipation"):
        super().__init__(name)

    @override
    def qubit(self, q) -> None:
        p = self.p_error
        if p > 0 and rng.random() < p:
            q.measure()
            assert q.state.num == 1
            q.state.rho = QUBIT_RHO_0

    @override
    def werner(self, q) -> None:
        """
        Dissipation in Werner state is approximated as a decay of visibility.
        """
        q.w *= self.p_survival

    @override
    def mixed(self, q) -> None:
        raise NotImplementedError()
