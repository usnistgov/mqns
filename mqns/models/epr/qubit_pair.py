from collections.abc import Iterable
from typing import Unpack, final, overload, override

import numpy as np

from mqns.models.core import BASIS_X, BASIS_Z, Basis
from mqns.models.core.state import BELL_RHO_PHI_P, qubit_rho_remove
from mqns.models.epr.entanglement import Entanglement, EntanglementInitKwargs, PurifProtocol
from mqns.models.error import PauliErrorModel, PerfectErrorModel
from mqns.models.qubit import CNOT, RX, H, QState, Qubit


@final
class EntangledQubitPair(Entanglement):
    """
    Entanglement model represented with two ``Qubit`` instances.

    This model is primarily intended for numeric verification of other models.
    """

    q0: Qubit
    q1: Qubit

    @overload
    def __init__(self, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Construct maximally entangled ``|Φ+>`` state.
        """

    @overload
    def __init__(self, q0: Qubit, q1: Qubit, /, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Construct from existing qubits.
        """

    def __init__(self, *qubits: Qubit, **kwargs: Unpack[EntanglementInitKwargs]):
        super().__init__(**kwargs)
        if qubits:
            QState.joint(*qubits)
            self.q0, self.q1 = qubits
        else:
            self.q0, self.q1 = Qubit.create_multi(2, rho=BELL_RHO_PHI_P)

    @staticmethod
    def move_from(epr: Entanglement) -> "EntangledQubitPair":
        """
        Convert an entanglement to ``EntangledQubitPair``.

        Args:
            epr: An entanglement using any model, which would become decohered
                 as its entanglement state is moved to the returned instance.

        Returns:
            ``EntangledQubitPair`` with equivalent entanglement state and fidelity.

            These attributes are copied:

            * decohere_time
            * fidelity_time
            * src, dst
            * mem_keys
            * store_decays

            These attributes are not copied:

            * is_decohered
            * orig_eprs
        """
        ne = EntangledQubitPair(
            *epr.to_qubits(),
            name=epr.name,
            decohere_time=epr.decohere_time,
            fidelity_time=epr.fidelity_time,
            src=epr.src,
            dst=epr.dst,
            mem_keys=epr.mem_keys,
            store_decays=epr.store_decays,
        )
        return ne

    @property
    @override
    def fidelity(self) -> float:
        if self.q0.state is not self.q1.state:
            assert self.is_decohered
            return 0.0

        state = self.q0.state
        rho = state.rho
        idx0 = state.qubits.index(self.q0)
        idx1 = state.qubits.index(self.q1)

        # Trace out other qubits (if any) from the density matrix.
        if (n := state.num) > 2:
            rho = qubit_rho_remove(rho, (i for i in range(n) if i not in (idx0, idx1)), n)

        # If the density matrix is in [q1,q0] order, swap the qubits.
        # Note that idx0,idx1 may not be the current indices, but the trace-out function preserves
        # their relative ordering, so that ``idx1<idx0`` condition is still correct.
        if idx1 < idx0:
            rho = rho.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)

        # Compare to the desired state BELL_RHO_PHI_P.
        f_val = 0.5 * (rho[0, 0] + rho[0, 3] + rho[3, 0] + rho[3, 3])
        return np.real(f_val)

    @fidelity.setter
    @override
    def fidelity(self, value: float):
        _ = value
        raise TypeError("cannot assign fidelity to EntangledQubitPair")

    @override
    def flip_direction(self) -> None:
        super().flip_direction()
        self.q0, self.q1 = self.q1, self.q0

    @staticmethod
    @override
    def _make_swapped(epr0: "EntangledQubitPair", epr1: "EntangledQubitPair", **kwargs: Unpack[EntanglementInitKwargs]):
        q3 = epr1.teleportation(epr0.q1)
        epr1.is_decohered = False  # .teleportation sets .is_decohered=True but Entanglement.swap() needs it False
        return EntangledQubitPair(epr0.q0, q3, **kwargs)

    @override
    def _do_purify(self, epr1: "EntangledQubitPair", protocol: PurifProtocol, basis: Basis) -> bool:
        # Rotate to X-basis if specified.
        if basis is BASIS_X:
            H(self.q0)
            H(self.q1)
            H(epr1.q0)
            H(epr1.q1)
        elif basis is not BASIS_Z:
            raise ValueError(f"cannot purify in {basis.name} basis")

        # Apply protocol-specific pre-rotations.
        if protocol is PurifProtocol.BBPSSW:
            pass
        elif protocol is PurifProtocol.DEJMPS:
            RX(self.q0, np.pi / 2)
            RX(self.q1, -np.pi / 2)
            RX(epr1.q0, np.pi / 2)
            RX(epr1.q1, -np.pi / 2)
        else:
            raise ValueError(f"cannot purify with {protocol} protocol")

        # Bilateral CNOT.
        CNOT(self.q0, epr1.q0)
        CNOT(self.q1, epr1.q1)

        # Apply protocol-specific post-rotations to the kept pair.
        if protocol is PurifProtocol.DEJMPS:
            RX(self.q0, -np.pi / 2)
            RX(self.q1, np.pi / 2)

        # Rotate back from X-basis for kept pair.
        if basis is BASIS_X:
            H(self.q0)
            H(self.q1)

        # Measure the consumed pair to determine whether success.
        m0 = epr1.q0.measure()
        m1 = epr1.q1.measure()
        return m0 == m1

    @override
    def apply_error(self, error) -> None:
        if isinstance(error, PauliErrorModel | PerfectErrorModel):
            self.q0.apply_error(error)
        else:
            raise TypeError(f"non-Pauli error {type(error)} must be applied to an individual qubit")

    @override
    def to_qubits(self) -> tuple[Qubit, Qubit]:
        if self.is_decohered:
            return super().to_qubits()
        self.is_decohered = True  # detaching the qubits
        return self.q0, self.q1

    @override
    def _describe_fidelity(self) -> Iterable[str]:
        state = self.q0.state
        yield f"rho={state.rho}"
        yield f"idx0={state.qubits.index(self.q0)}"
        yield f"idx1={state.qubits.index(self.q1)}"
