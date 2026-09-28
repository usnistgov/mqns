from collections.abc import Iterable
from typing import Unpack, final, overload, override

from mqns.models.core import BASIS_X, BASIS_Z, Basis
from mqns.models.core.bell_diagonal import (
    BellDiagonalProbV,
    bell_diagonal_probv_to_pauli_transfer_mat,
    make_bell_diagonal_probv,
    normalize_bell_diagonal_probv,
)
from mqns.models.core.state import (
    ATOL,
    BELL_RHO_PHI_N,
    BELL_RHO_PHI_P,
    BELL_RHO_PSI_N,
    BELL_RHO_PSI_P,
    QubitRho,
    check_qubit_rho,
)
from mqns.models.epr.entanglement import Entanglement, EntanglementInitKwargs, PurifProtocol
from mqns.utils import rng

_probv_1 = make_bell_diagonal_probv(1, 0, 0, 0)


@final
class MixedStateEntanglement(Entanglement):
    """
    Bell-Diagonal state entanglement model.
    """

    @overload
    def __init__(self, *, fidelity=1.0, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Construct with fidelity.

        This creates a Werner state where the error probabilities are distributed equally among the three bad states.
        """

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

    @overload
    def __init__(self, *, probv: BellDiagonalProbV, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Construct with probability vector.

        Args:
            probv: Probability vector.
        """

    def __init__(
        self,
        *,
        probv: BellDiagonalProbV | None = None,
        fidelity: float | None = None,
        i: float | None = None,
        z=0.0,
        x=0.0,
        y=0.0,
        **kwargs: Unpack[EntanglementInitKwargs],
    ):
        super().__init__(**kwargs)
        if probv is not None:
            self.set_probv(probv)
        elif i is not None:
            self.set_probv(make_bell_diagonal_probv(i, z, x, y), normalize=False)
        elif fidelity is not None:
            self.fidelity = fidelity
        else:
            self.set_probv(_probv_1, normalize=False)

    @property
    @override
    def fidelity(self) -> float:
        return self.probv[0]

    @fidelity.setter
    @override
    def fidelity(self, value: float):
        """Reset fidelity, turning into a Werner state."""
        zxy = (1 - value) / 3
        self.set_probv(make_bell_diagonal_probv(value, zxy, zxy, zxy), normalize=False)

    @property
    def probv(self) -> BellDiagonalProbV:
        """Probability vector: I,Z,X,Y."""
        return self._probv

    def set_probv(self, probv: BellDiagonalProbV, *, normalize=True, copy=True) -> None:
        """
        Update probability vector.

        Args:
            probv: New probability vector.
            normalize: If False, assume ``probv`` is already normalized.
            copy: If False, ``probv`` may be normalized in-place.
        """
        if normalize:
            if copy:
                probv = probv.copy()
            probv = normalize_bell_diagonal_probv(probv)
        self._probv = probv

    @staticmethod
    @override
    def _make_swapped(epr0: "MixedStateEntanglement", epr1: "MixedStateEntanglement", **kwargs: Unpack[EntanglementInitKwargs]):
        return MixedStateEntanglement(probv=bell_diagonal_probv_to_pauli_transfer_mat(epr0.probv) @ epr1.probv, **kwargs)

    @override
    def _do_purify(self, epr1: "MixedStateEntanglement", protocol: PurifProtocol, basis: Basis) -> bool:
        if basis not in (BASIS_Z, BASIS_X):
            raise ValueError(f"cannot purify in {basis.name} basis")

        i0, z0, x0, y0 = self.probv
        i1, z1, x1, y1 = epr1.probv

        if protocol is PurifProtocol.BBPSSW:
            p_succ = (i0 + z0) * (i1 + z1) + (x0 + y0) * (x1 + y1)
            if basis is BASIS_Z:
                new_i = i0 * i1 + z0 * z1
                new_z = z0 * i1 + i0 * z1
                new_x = x0 * x1 + y0 * y1
                new_y = x0 * y1 + y0 * x1
            else:
                new_i = i0 * i1 + x0 * x1
                new_z = z0 * z1 + y0 * y1
                new_x = x0 * i1 + i0 * x1
                new_y = z0 * y1 + y0 * z1
        elif protocol is PurifProtocol.DEJMPS:
            p_succ = (i0 + y0) * (i1 + y1) + (z0 + x0) * (x1 + z1)
            new_i = i0 * i1 + y0 * y1
            new_y = i0 * y1 + y0 * i1
            if basis is BASIS_Z:
                new_z = z0 * z1 + x0 * x1
                new_x = z0 * x1 + x0 * z1
            else:
                new_z = x0 * z1 + z0 * x1
                new_x = x0 * x1 + z0 * z1
        else:
            raise ValueError(f"cannot purify with {protocol} protocol")

        if p_succ <= ATOL or rng.random() > p_succ:
            return False

        self.set_probv(make_bell_diagonal_probv(new_i, new_z, new_x, new_y), normalize=False)
        return True

    @override
    def apply_error(self, error) -> None:
        error.mixed(self)

    @override
    def _to_qubits_rho(self) -> QubitRho:
        i, z, x, y = self.probv
        return check_qubit_rho(i * BELL_RHO_PHI_P + z * BELL_RHO_PHI_N + x * BELL_RHO_PSI_P + y * BELL_RHO_PSI_N, n=2)

    @override
    def _describe_fidelity(self) -> Iterable[str]:
        i, z, x, y = self.probv
        yield f"i={i:.4f}"
        yield f"z={z:.4f}"
        yield f"x={x:.4f}"
        yield f"y={y:.4f}"
