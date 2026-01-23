#    Modified by Amar Abane for Multiverse Quantum Network Simulator
#    Date: 05/17/2025
#    Summary of changes: Adapted logic to support dynamic approaches.
#
#    This file is based on a snapshot of SimQN (https://github.com/QNLab-USTC/SimQN),
#    which is licensed under the GNU General Public License v3.0.
#
#    The original SimQN header is included below.


#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import hashlib
import uuid
from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar, Unpack, cast

import numpy as np

from mqns.models.core import QuantumModel
from mqns.models.core.operator import OPERATOR_PAULI_I, Operator
from mqns.models.core.state import QUBIT_STATE_P, QubitRho, build_qubit_state, qubit_state_to_rho
from mqns.models.qubit import QState, Qubit
from mqns.models.qubit.gate import CNOT, H, U, X, Y, Z
from mqns.simulator import Time
from mqns.utils import rng

if TYPE_CHECKING:
    from mqns.entity.node import QNode


EntanglementT = TypeVar("EntanglementT", bound="Entanglement")

EntanglementStoreErrorFunc = Callable[["Entanglement", Time], None] | None


class EntanglementInitKwargs(TypedDict, total=False):
    name: str | None
    creation_time: Time
    decoherence_time: Time
    src: "QNode|None"
    dst: "QNode|None"
    store_errors: tuple[EntanglementStoreErrorFunc, EntanglementStoreErrorFunc]


class Entanglement(Generic[EntanglementT], QuantumModel):
    """Base entanglement model."""

    def __init__(self, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Constructor.

        Args:
            name: Entanglement name, defaults to a random string.
            creation_time: EPR creation time point, defaults to `Time.SENTINEL`.
            decoherence_time: Qubits decoherence time point, defaults to `Time.SENTINEL`.
            src: Left node that holds one entangled qubit.
            dst: Right node that holds one entangled qubit.
            store_errors: Memory store error function at src and dst.
        """
        name = kwargs.get("name")
        self.name = uuid.uuid4().hex if name is None else name
        """Descriptive name."""

        self.is_decoherenced = False
        """Whether the entanglement has decohered."""
        self.creation_time = kwargs.get("creation_time", Time.SENTINEL)
        """
        EPR creation time point assigned by LinkLayer or swapping.
        Some operations are unavailable if this is `Time.SENTINEL`.
        """
        self.decoherence_time = kwargs.get("decoherence_time", Time.SENTINEL)
        """
        EPR decoherence time point assigned by memory or swapping.
        Some operations are unavailable if this is `Time.SENTINEL`.
        """
        self.read = False
        """Whether the entanglement has been read from the memory by either node."""

        self.key: str | None = None
        """Reservation key used by LinkLayer."""
        self.src = kwargs.get("src")
        """One node that holds one entangled qubit, at the left side of a path."""
        self.dst = kwargs.get("dst")
        """The other node that holds the other entangled qubit, at the right side of a path."""
        self.store_errors = kwargs.get("store_errors", (None, None))
        """Memory decoherence rate in Hz at src and dst."""

        self.ch_index = -1
        """
        Index of elementary entanglement in a path, smaller indices are on the left side.
        Negative means this is not an elementary entanglement.
        """
        self.orig_eprs: list[EntanglementT] = []
        """Elementary entanglements that swapped into this entanglement."""
        self.tmp_path_ids: frozenset[int] | None = None
        """Possible path IDs, used by MuxSchemeStatistical and MuxSchemeDynamicEpr."""

    @property
    @abstractmethod
    def fidelity(self) -> float:
        pass

    @fidelity.setter
    @abstractmethod
    def fidelity(self, value: float):
        pass

    def _mark_decoherenced(self) -> None:
        """Mark the EPR as decoherenced."""
        self.is_decoherenced = True

    def apply_store_errors(self, now: Time) -> None:
        """Apply store error models for both qubits in this EPR."""
        if self.read:
            return
        t = now - self.creation_time
        for se in self.store_errors:
            if se is not None:
                se(self, t)
        self.read = True

    @staticmethod
    def swap(
        epr0: EntanglementT,
        epr1: EntanglementT,
        *,
        now: Time,
        ps=1.0,
    ) -> EntanglementT | None:
        """
        Perform swapping between `epr0` and `epr1`, and distribute a new entanglement.

        Args:
            epr0: left entanglement.
            epr1: right entanglement.
            now: current timestamp.
            ps: probability of successful swapping.

        Returns:
            New entanglement, or None if swap failed.
        """

        assert type(epr0) is type(epr1)
        assert epr0.dst == epr1.src  # it's okay for src and dst to be None

        if epr0.is_decoherenced or epr1.is_decoherenced:
            epr0._mark_decoherenced()
            epr1._mark_decoherenced()
            return None

        if ps < 1.0 and rng.random() >= ps:  # swap failed
            epr0._mark_decoherenced()
            epr1._mark_decoherenced()
            return None

        orig_eprs: list[EntanglementT] = []
        for epr in (epr0, epr1):
            if not epr.read:
                epr.apply_store_errors(now)

            if epr.ch_index > -1:
                orig_eprs.append(epr)
            else:
                orig_eprs.extend(cast(list[EntanglementT], epr.orig_eprs))

        orig_names = "-".join((e.name for e in orig_eprs))
        name = hashlib.sha256(orig_names.encode()).hexdigest()[:32]  # same length as `uuid.uuid4().hex`
        ne = type(epr0)._make_swapped(
            epr0,
            epr1,
            name=name,
            creation_time=now,
            decoherence_time=min(epr0.decoherence_time, epr1.decoherence_time),
            src=epr0.src,
            dst=epr1.dst,
            store_errors=(epr0.store_errors[0], epr1.store_errors[1]),
        )
        ne.orig_eprs = orig_eprs
        return ne

    @staticmethod
    @abstractmethod
    def _make_swapped(epr0: EntanglementT, epr1: EntanglementT, **kwargs: Unpack[EntanglementInitKwargs]) -> EntanglementT:
        """
        Create a new entanglement that is the result of successful swapping between epr0 and epr1.
        Most properties are provided in kwargs or assigned subsequently.
        Subclass implementation should calculate the fidelity of new entanglement.
        """
        pass

    def purify(self, epr1: EntanglementT, *, now: Time) -> bool:
        """
        Perform purification on `self` consuming `epr1`.

        Args:
            self: kept entanglement.
            epr1: consumed entanglement.
            now: current timestamp.

        Returns:
            Whether successful.
        """

        assert type(self) is type(epr1)
        assert (self.src, self.dst) == (epr1.src, epr1.dst)  # it's okay for src and dst to be None

        if self.is_decoherenced or epr1.is_decoherenced:
            self._mark_decoherenced()
            epr1._mark_decoherenced()
            return False

        _ = now
        ok = self._do_purify(epr1)

        if not ok:
            self._mark_decoherenced()
        epr1._mark_decoherenced()

        return ok

    @abstractmethod
    def _do_purify(self, epr1: EntanglementT) -> bool:
        pass

    def to_qubits(self) -> tuple[Qubit, Qubit]:
        """
        Transport the entanglement into a pair of qubits based on the fidelity.
        Maximally entanglement returns ``|Φ+>`` state.

        Returns:
            A tuple of two qubits.
        """
        if self.is_decoherenced:
            q0 = Qubit(QUBIT_STATE_P, name="q0")
            q1 = Qubit(QUBIT_STATE_P, name="q1")
            return (q0, q1)

        q0 = Qubit(name="q0")
        q1 = Qubit(name="q1")
        qs = QState([q0, q1], rho=self._to_qubits_rho())
        q0.state = qs
        q1.state = qs

        self.is_decoherenced = True
        return (q0, q1)

    def _to_qubits_rho(self) -> QubitRho:
        a = np.sqrt(self.fidelity / 2)
        b = np.sqrt((1 - self.fidelity) / 2)
        state = build_qubit_state((a, b, b, a), 2)
        return qubit_state_to_rho(state, 2)

    def teleportation(self, qubit: Qubit) -> Qubit:
        """
        Use `self` and `qubit` to perform teleportation.
        """
        q1, q2 = self.to_qubits()
        CNOT(qubit, q1)
        H(qubit)
        c0 = qubit.measure()
        c1 = q1.measure()
        if c1 == 1 and c0 == 0:
            X(q2)
        elif c1 == 0 and c0 == 1:
            Z(q2)
        elif c1 == 1 and c0 == 1:
            Y(q2)
            U(q2, Operator(np.complex128(1j) * OPERATOR_PAULI_I.u, 1))
        self.is_decoherenced = True
        return q2

    def __repr__(self) -> str:
        return ", ".join(_describe(self)) + ")"

    def _describe_fidelity(self) -> Iterable[str]:
        yield f"fidelity={self.fidelity:.4f}"


def _describe(epr: Entanglement) -> Iterable[str]:
    yield f"EPR({epr.name}"

    if epr.is_decoherenced:
        yield "DECOHERENCED"
    else:
        yield from epr._describe_fidelity()

    if epr.creation_time is not Time.SENTINEL:
        yield f"creation_time={epr.creation_time.sec}"
    if epr.decoherence_time is not Time.SENTINEL:
        yield f"decoherence_time={epr.decoherence_time.sec}"

    if epr.src and epr.dst:
        yield f"src={epr.src.name}"
        yield f"dst={epr.dst.name}"
        if epr.ch_index >= 0:
            yield f"ch_index={epr.ch_index}"
        elif len(epr.orig_eprs) > 0:
            orig_eprs = ",".join(e.name for e in epr.orig_eprs)
            yield f"orig_eprs=[{orig_eprs}]"

    if epr.tmp_path_ids:
        tmp_path_ids = ",".join(str(x) for x in epr.tmp_path_ids)
        yield f"tmp_path_ids=[{tmp_path_ids}]"
