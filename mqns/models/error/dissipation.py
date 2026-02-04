from typing import override

from mqns.models.core.bell_diagonal import make_bell_diagonal_probv
from mqns.models.core.state import QUBIT_STATE_0
from mqns.models.error.pauli import PauliErrorModelBase
from mqns.utils import rng


class DissipationErrorModel(PauliErrorModelBase):
    """
    Dissipation error model: qubit becomes ``|0>`` with ``p_error`` probability.

    Single qubit:
        Affected qubit decays to the ground state.
        Partner qubit collapses into a mixed state.

    Werner state:
        Dissipation is approximated as a decay of visibility.

    Bell-diagonal state:
        Bell-diagonal state is always unbiased, whereas dissipation introduces a bias toward ``|0>``.
        To stay with Bell-diagonal, the resulting state is twirled back into a Bell-diagonal form.
        Dissipation is approximated as a combination of Pauli errors: a 50/50 mixture of bip-flip and phase-flip.
    """

    def __init__(self, name="dissipation"):
        super().__init__(name)

    @override
    def _prepare(self) -> None:
        p = self.p_error
        if p <= 0:
            self._set_probv(self._probv0)
        else:
            p2 = p / 2
            self._set_probv(make_bell_diagonal_probv(1 - p2, p2, 0, 0))

    @override
    def qubit(self, q) -> None:
        p = self.p_error
        if p > 0 and rng.random() < p:
            q.state.trace_out(q, QUBIT_STATE_0)
