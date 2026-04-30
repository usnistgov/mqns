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
from collections.abc import Iterable
from typing import TYPE_CHECKING, Self, TypedDict, Unpack, cast

import numpy as np

from mqns.models.core import QuantumModel
from mqns.models.core.operator import OPERATOR_PAULI_I, Operator
from mqns.models.core.state import QUBIT_STATE_P, QubitRho, build_qubit_state, qubit_state_to_rho
from mqns.models.error import ErrorModel, PerfectErrorModel, TimeDecayFunc, time_decay_nop
from mqns.models.qubit import QState, Qubit
from mqns.models.qubit.gate import CNOT, H, U, X, Y, Z
from mqns.simulator import Time
from mqns.utils import rng

if TYPE_CHECKING:
    from mqns.entity.node import QNode


class EntanglementInitKwargs(TypedDict, total=False):
    name: str | None
    decohere_time: Time
    fidelity_time: Time
    src: "QNode|None"
    dst: "QNode|None"
    store_decays: tuple[TimeDecayFunc | None, TimeDecayFunc | None]


class Entanglement(QuantumModel):
    """Base entanglement model."""

    is_decohered: bool = False
    """
    Whether the entanglement has decohered.

    This reflects the hidden physical state.
    It must not be used to guide decision prior to measurement.
    """

    read: bool = False
    """
    Whether the entanglement has been read from the memory by either node.

    Note: This is a legacy attribute, planned for removal.
    """

    key: str | None = None
    """Reservation key used by LinkLayer."""

    ch_index: int = -1
    """
    Index of elementary entanglement in a path, smaller indices are on the left side.
    Negative means this is not an elementary entanglement.
    """
    orig_eprs: list[Self] | None = None
    """Elementary entanglements that swapped into this entanglement."""
    tmp_path_ids: frozenset[int] | None = None
    """Possible path IDs, used by MuxSchemeStatistical and MuxSchemeDynamicEpr."""

    def __init__(self, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Constructor.

        Args:
            name: Entanglement name, defaults to a random string.
            decohere_time: EPR decoherence time point, defaults to ``Time.SENTINEL``.
            fidelity_time: EPR creation or fidelity update time point, defaults to ``Time.SENTINEL``.
            src: Left node that holds one of the entangled qubits.
            dst: Right node that holds one of the entangled qubits.
            store_decays: Memory time-based decay functions at src and dst.
        """
        name = kwargs.get("name")
        self.name = uuid.uuid4().hex if name is None else name
        """Descriptive name."""

        self.decohere_time = kwargs.get("decohere_time", Time.SENTINEL)
        """
        EPR decoherence time point, when it would be removed from memory.

        * Upon creating a memory-memory EPR, this is assigned according to memory ``t_cohere``.
          If the two qubits are stored in memories with different dephasing times, the shorter value is used.
        * Upon swapping, the oldest decoherence time point is used.
        * Upon purification, the decoherence time point is unchanged.
        * Some operations are unavailable if this is ``Time.SENTINEL``.
        """
        self.fidelity_time = kwargs.get("fidelity_time", Time.SENTINEL)
        """
        Fidelity update time point, reflecting when fidelity values are last updated.

        * Upon creating a memory-memory EPR, this is assigned according to the EPR creation time.
        * This may be updated to the current time at any time, while ``store_decays`` is applied to the EPR.
        * Some operations are unavailable if this is ``Time.SENTINEL``.
        """

        self.src = kwargs.get("src")
        """Left node that holds one of the entangled qubits."""
        self.dst = kwargs.get("dst")
        """Right node that holds one of the entangled qubits."""
        decay0, decay1 = kwargs.get("store_decays", (None, None))
        self.store_decays = (decay0 or time_decay_nop, decay1 or time_decay_nop)
        """Memory time-based decay functions at src and dst."""

    @property
    @abstractmethod
    def fidelity(self) -> float:
        """
        Calculate fidelity.

        This value is valid only if ``is_decohered`` is False.
        """

    @fidelity.setter
    @abstractmethod
    def fidelity(self, value: float):
        pass

    def apply_store_decays(self, now: Time) -> None:
        """
        Apply memory time-based decays for both qubits in this EPR.

        Args:
            now: Current time point.
        """
        t = now - self.fidelity_time
        if self.read or t.time_slot == 0:
            return
        for se in self.store_decays:
            se(self, t)
        self.fidelity_time = now
        self.read = True

    @staticmethod
    def swap[E: Entanglement](
        epr0: E,
        epr1: E,
        *,
        now: Time,
        ps=1.0,
        error: ErrorModel = PerfectErrorModel(),
    ) -> tuple[E, bool]:
        """
        Perform swapping between ``epr0`` and ``epr1``, and distribute a new entanglement.

        Args:
            epr0: Left entanglement.
            epr1: Right entanglement.
            now: Current timestamp.
            ps: Probability of successful swapping.
            error: BSA error model.

        Returns:
            [0]: New entanglement.
            [1]: Whether success from local point of view.
        """

        assert type(epr0) is type(epr1)
        assert epr0.dst == epr1.src  # it's okay for src and dst to be None

        orig_eprs: list[E] = []
        for epr in (epr0, epr1):
            epr.apply_store_decays(now)
            if epr.orig_eprs:
                orig_eprs.extend(cast(list[E], epr.orig_eprs))
            else:
                orig_eprs.append(epr)

        orig_names = "-".join((e.name for e in orig_eprs))
        name = hashlib.sha256(orig_names.encode()).hexdigest()[:32]  # same length as `uuid.uuid4().hex`
        ne = cast(
            E,
            type(epr0)._make_swapped(
                epr0,
                epr1,
                name=name,
                decohere_time=min(epr0.decohere_time, epr1.decohere_time),
                fidelity_time=now,
                src=epr0.src,
                dst=epr1.dst,
                store_decays=(epr0.store_decays[0], epr1.store_decays[1]),
            ),
        )
        ne.orig_eprs = orig_eprs

        local_failure = ps < 1.0 and rng.random() >= ps
        ne.is_decohered = epr0.is_decohered or epr1.is_decohered or local_failure
        if not ne.is_decohered:
            ne.apply_error(error)
        return ne, not local_failure

    @staticmethod
    @abstractmethod
    def _make_swapped(epr0, epr1, **kwargs: Unpack[EntanglementInitKwargs]) -> "Entanglement":
        """
        Create a new entanglement that is the result of successful swapping between epr0 and epr1.
        Most properties are provided in kwargs or assigned subsequently.
        Subclass implementation should calculate the fidelity of new entanglement.
        """

    def purify(self, epr1: Self, *, now: Time) -> bool:
        """
        Perform purification on ``self`` consuming ``epr1``.

        Args:
            self: kept entanglement.
            epr1: consumed entanglement.
            now: current timestamp.

        Returns:
            Whether successful.
        """

        assert type(self) is type(epr1)
        assert (self.src, self.dst) == (epr1.src, epr1.dst)  # it's okay for src and dst to be None

        if self.is_decohered or epr1.is_decohered:
            return False

        _ = now
        ok = self._do_purify(epr1)

        if not ok:
            self.is_decohered = True
        epr1.is_decohered = True

        return ok

    @abstractmethod
    def _do_purify(self, epr1) -> bool:
        pass

    def to_qubits(self) -> tuple[Qubit, Qubit]:
        """
        Transport the entanglement into a pair of qubits based on the fidelity.
        Maximally entanglement returns ``|Φ+>`` state.

        Returns:
            A tuple of two qubits.
        """
        if self.is_decohered:
            q0 = Qubit(QUBIT_STATE_P, name="q0")
            q1 = Qubit(QUBIT_STATE_P, name="q1")
            return (q0, q1)

        q0 = Qubit(name="q0")
        q1 = Qubit(name="q1")
        qs = QState([q0, q1], rho=self._to_qubits_rho())
        q0.state = qs
        q1.state = qs

        self.is_decohered = True
        return (q0, q1)

    def _to_qubits_rho(self) -> QubitRho:
        a = np.sqrt(self.fidelity / 2)
        b = np.sqrt((1 - self.fidelity) / 2)
        state = build_qubit_state((a, b, b, a), 2)
        return qubit_state_to_rho(state, 2)

    def teleportation(self, qubit: Qubit) -> Qubit:
        """
        Use ``self`` and ``qubit`` to perform teleportation.
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
        self.is_decohered = True
        return q2

    def __repr__(self) -> str:
        return ", ".join(_describe(self)) + ")"

    def _describe_fidelity(self) -> Iterable[str]:
        yield f"fidelity={self.fidelity:.4f}"


def _describe(epr: Entanglement) -> Iterable[str]:
    yield f"EPR({epr.name}"

    if epr.is_decohered:
        yield "DECOHERED"
    else:
        yield from epr._describe_fidelity()

    if epr.fidelity_time is not Time.SENTINEL:
        yield f"fidelity_time={epr.fidelity_time.sec}"
    if epr.decohere_time is not Time.SENTINEL:
        yield f"decohere_time={epr.decohere_time.sec}"

    if epr.src and epr.dst:
        yield f"src={epr.src.name}"
        yield f"dst={epr.dst.name}"
        if epr.ch_index >= 0:
            yield f"ch_index={epr.ch_index}"
        elif epr.orig_eprs:
            orig_eprs = ",".join(e.name for e in epr.orig_eprs)
            yield f"orig_eprs=[{orig_eprs}]"

    if epr.tmp_path_ids:
        tmp_path_ids = ",".join(str(x) for x in epr.tmp_path_ids)
        yield f"tmp_path_ids=[{tmp_path_ids}]"
