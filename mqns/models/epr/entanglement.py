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
from abc import abstractmethod
from collections.abc import Iterable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Self, TypedDict, Unpack, cast

import numpy as np

from mqns.models.core import BASIS_Z, Basis, QuantumModel
from mqns.models.core.state import QUBIT_STATE_P, QubitRho, build_qubit_state, qubit_state_to_rho
from mqns.models.error import ErrorModel, PerfectErrorModel, TimeDecayFunc, time_decay_nop
from mqns.models.qubit import CNOT, H, Qubit, X, Z
from mqns.simulator import Time
from mqns.utils import AutoIncrementIdentifier, rng

if TYPE_CHECKING:
    from mqns.entity.node import QNode


_AUTOID = AutoIncrementIdentifier("etg_")
"""
Automatically assigned ``Entanglement.name`` generator.
"""


class EntanglementInitKwargs(TypedDict, total=False):
    name: str | None
    decohere_time: Time
    fidelity_time: Time
    src: "QNode|None"
    dst: "QNode|None"
    mem_keys: tuple[str | None, str | None]
    store_decays: tuple[TimeDecayFunc | None, TimeDecayFunc | None]


class PurifProtocol(Enum):
    BBPSSW = auto()
    DEJMPS = auto()


class Entanglement(QuantumModel):
    """Base entanglement model."""

    name: str
    """Descriptive name."""

    is_decohered: bool = False
    """
    Whether the entanglement has decohered.

    This reflects the hidden physical state.
    It must not be used to guide decision prior to measurement.
    """
    decohere_time: Time
    """
    EPR decoherence time point, when it would be removed from memory.

    * Upon creating a memory-memory EPR, this is assigned according to memory ``t_cohere``.
        If the two qubits are stored in memories with different dephasing times, the shorter value is used.
    * Upon swapping, the oldest decoherence time point is used.
    * Upon purification, the decoherence time point is unchanged.
    * Some operations are unavailable if this is ``Time.SENTINEL``.
    """
    fidelity_time: Time
    """
    Fidelity update time point, reflecting when fidelity values are last updated.

    * Upon creating a memory-memory EPR, this is assigned according to the EPR creation time.
    * This may be updated to the current time at any time, while ``store_decays`` is applied to the EPR.
    * Some operations are unavailable if this is ``Time.SENTINEL``.
    """

    src: "QNode|None"
    """Left node that holds one of the entangled qubits."""
    dst: "QNode|None"
    """Right node that holds one of the entangled qubits."""
    mem_keys: tuple[str, str]
    """MemoryQubit reservation keys at src and dst."""
    store_decays: tuple[TimeDecayFunc, TimeDecayFunc]
    """Memory time-based decay functions at src and dst."""

    orig_eprs: list[Self] | None = None
    """Elementary entanglements that swapped into this entanglement."""
    affectionated_path_id: int = -1
    """Path ID chosen by MuxSchemeDynamicEpr or MuxSchemeStatistical(coordinated_decisions=True)."""
    consumed_sides: int = 0b00
    """
    Whether the entanglement has been consumed via ``consume_with_store_decay_side()`` on each side, as bitmap:
    * 0b01: Left side.
    * 0b10: Right side.
    """

    def __init__(self, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Constructor.

        Args:
            name: Entanglement name, defaults to a random string.
            decohere_time: EPR decoherence time point, defaults to ``Time.SENTINEL``.
            fidelity_time: EPR creation or fidelity update time point, defaults to ``Time.SENTINEL``.
            src: Left node that holds one of the entangled qubits.
            dst: Right node that holds one of the entangled qubits.
            mem_keys: MemoryQubit reservation keys at src and dst.
            store_decays: Memory time-based decay functions at src and dst.
        """
        name = kwargs.get("name")
        self.name = _AUTOID() if name is None else name

        self.decohere_time = kwargs.get("decohere_time", Time.SENTINEL)
        self.fidelity_time = kwargs.get("fidelity_time", Time.SENTINEL)

        self.src = kwargs.get("src")
        self.dst = kwargs.get("dst")
        key0, key1 = kwargs.get("mem_keys", (None, None))
        self.mem_keys = key0 or "", key1 or ""
        """MemoryQubit reservation keys at src and dst."""
        decay0, decay1 = kwargs.get("store_decays", (None, None))
        self.store_decays = decay0 or time_decay_nop, decay1 or time_decay_nop

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

    def flip_direction(self) -> None:
        """
        Reverse the direction between src and dst.
        """
        self.src, self.dst = self.dst, self.src
        self.mem_keys = self.mem_keys[1], self.mem_keys[0]
        self.store_decays = self.store_decays[1], self.store_decays[0]
        if self.orig_eprs:
            self.orig_eprs.reverse()
            for oe in self.orig_eprs:
                oe.flip_direction()
        self.consumed_sides = ((self.consumed_sides & 0b01) << 1) | ((self.consumed_sides & 0b10) >> 1)

    def apply_store_decays(self, now: Time) -> None:
        """
        Apply memory time-based decays for both qubits in this EPR.

        Args:
            now: Current time point.
        """
        t = now - self.fidelity_time
        assert self.consumed_sides == 0b00
        if t.slot == 0:
            return
        for se in self.store_decays:
            se(self, t)
        self.fidelity_time = now

    def consume_with_store_decay_side(self, now: Time, *, side: int) -> bool:
        """
        Apply memory time-based decays for one qubit in this EPR, when it is ready for consumption.

        Args:
            now: Current time point.
            side: 0 for left side, 1 for right side.

        Returns: Whether both qubits are consumed.

        This differs from ``apply_store_decays()`` in that:
        * ``fidelity_time`` is not updated, so that the other side can update from the same origin time.
        * ``consumed_sides`` indicates which side(s) have performed this update.
        """
        side_bit = 1 << side
        assert (self.consumed_sides & side_bit) == 0b00
        self.store_decays[side](self, now - self.fidelity_time)
        self.consumed_sides |= side_bit
        return self.consumed_sides == 0b11

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
            now: Timestamp to stop accumulating errors on ``epr0`` and ``epr1``,
                and start accumulating errors on new entanglement.
            ps: Probability of successful swapping.
            error: BSA error model.

        Returns:
            [0]: New entanglement.
            [1]: Whether success from local point of view.
        """

        assert type(epr0) is type(epr1)
        assert epr0.dst is epr1.src  # it's okay for src and dst to be None

        orig_eprs: list[E] = []
        for epr in epr0, epr1:
            epr.apply_store_decays(now)
            if epr.orig_eprs:
                orig_eprs.extend(cast(list[E], epr.orig_eprs))
            else:
                orig_eprs.append(epr)

        orig_names = "-".join(e.name for e in orig_eprs)
        name = hashlib.sha256(orig_names.encode()).hexdigest()[:32]  # same length as default name
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
                mem_keys=(epr0.mem_keys[0], epr1.mem_keys[1]),
                store_decays=(epr0.store_decays[0], epr1.store_decays[1]),
            ),
        )
        ne.orig_eprs = orig_eprs

        local_failure = ps < 1.0 and rng.random() >= ps
        ne.is_decohered = epr0.is_decohered or epr1.is_decohered or local_failure
        if not ne.is_decohered:
            ne.apply_error(error)
        epr0.is_decohered = epr1.is_decohered = True
        return ne, not local_failure

    @staticmethod
    @abstractmethod
    def _make_swapped(epr0, epr1, **kwargs: Unpack[EntanglementInitKwargs]) -> "Entanglement":
        """
        Create a new entanglement that is the result of successful swapping between epr0 and epr1.
        Most properties are provided in kwargs or assigned subsequently.
        Subclass implementation should calculate the fidelity of new entanglement.
        """

    def purify(
        self,
        epr1: Self,
        *,
        now: Time,
        protocol: PurifProtocol = PurifProtocol.DEJMPS,
        basis: Basis = BASIS_Z,
    ) -> bool:
        """
        Perform purification on ``self`` consuming ``epr1``.

        Args:
            self: Kept entanglement.
            epr1: Consumed entanglement.
            now: Current timestamp.
            protocol: Distillation protocol, either BBPSSW or DEJMPS.
            basis: Measurement basis, either ``BASIS_Z`` or ``BASIS_X``.

        Returns:
            Whether successful.
        """

        assert type(self) is type(epr1)
        assert (self.src, self.dst) == (epr1.src, epr1.dst)  # it's okay for src and dst to be None

        if self.is_decohered or epr1.is_decohered:
            return False

        for epr in self, epr1:
            epr.apply_store_decays(now)

        ok = self._do_purify(epr1, protocol, basis)

        if not ok:
            self.is_decohered = True
        epr1.is_decohered = True

        return ok

    @abstractmethod
    def _do_purify(self, epr1: Any, protocol: PurifProtocol, basis: Basis) -> bool:
        pass

    def to_qubits(self) -> tuple[Qubit, Qubit]:
        """
        Transport the entanglement into a pair of qubits.
        Maximal entanglement returns ``|Φ+>`` state.

        Post-condition:

        * The entanglement is marked as decohered.

        Returns:
            A tuple of two qubits.
        """
        if self.is_decohered:
            q0 = Qubit(QUBIT_STATE_P)
            q1 = Qubit(QUBIT_STATE_P)
        else:
            q0, q1 = Qubit.create_multi(2, rho=self._to_qubits_rho())
            self.is_decohered = True
        return q0, q1

    def _to_qubits_rho(self) -> QubitRho:
        a = np.sqrt(self.fidelity / 2)
        b = np.sqrt((1 - self.fidelity) / 2)
        state = build_qubit_state((a, b, b, a), 2)
        return qubit_state_to_rho(state, 2)

    def teleportation(self, qubit: Qubit) -> Qubit:
        """
        Teleport the state of ``qubit`` to the remote half of this entangled pair.

        Args:
            qubit: The qubit whose state is to be teleported.
                   It shall be co-located with the first qubit in ``self.to_qubits()``.

        Post-condition and return value:

        * ``qubit`` is consumed.
        * The entanglement is consumed / decohered.
        * Returned qubit has same state as input ``qubit``.
          It is co-located with the second qubit from ``self.to_qubits()``.
        """
        q0 = qubit
        q1, q2 = self.to_qubits()

        # Perform Bell state measurement on the local half.
        CNOT(q0, q1)
        H(q0)
        m0 = q0.measure()
        m1 = q1.measure()

        # Apply Pauli corrections on the remote half.
        if m1 == 1:
            X(q2)
        if m0 == 1:
            Z(q2)

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
        if epr.orig_eprs:
            yield f"orig_eprs=[{','.join(e.name for e in epr.orig_eprs)}]"

    if epr.affectionated_path_id >= 0:
        yield f"affectionated_path_id={epr.affectionated_path_id}"
