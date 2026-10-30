from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, Self

import numpy as np

from mqns.entity.node import QNode
from mqns.simulator import Time
from mqns.utils import json_encodable

if TYPE_CHECKING:
    from mqns.network.ll.task import EntangleTask


@json_encodable
class LinkLayerCounters:
    @staticmethod
    def aggregate(nodes: Iterable[QNode]) -> "LinkLayerCounters":
        """
        Aggregate ``LinkLayerCounters`` from a network.

        Args:
            nodes: List of nodes, such as ``QuantumNetwork.nodes``.

        These fields cannot be aggregated:

        * ``etg_rate``
        """
        from mqns.network.ll.linklayer import LinkLayer  # noqa: PLC0415

        r = LinkLayerCounters()
        for node in nodes:
            for ll in node.get_apps(LinkLayer):
                r += ll.cnt
        return r

    n_etg: int = 0
    """How many entanglements generated as the primary node."""

    n_attempts: int = 0
    """How many attempts made for successful entanglements."""

    n_decoh: int = 0
    """How many qubits decohered."""

    def __init__(self, *, history=0, rtt=Time.SENTINEL):
        """
        Args:
            history: How many recent entanglement to keep in history.
                This must be positive for ``etg_rate`` calculation.
            rtt: Round-trip duration for RESERVE_REQ and RESERVE_RES messages.
                This is considered part of the entanglement generation duration.
        """
        # _history is a count-based window that tracks a variable duration of recent entanglements.
        # If the quantum channel properties could change over time, this may need to be refactored as
        # a time-based window or a time decay formula.
        self._history = deque[int](maxlen=history) if history > 0 and rtt is not Time.SENTINEL else None
        self._rtt = rtt

    def __iadd__(self, cnt: "LinkLayerCounters") -> Self:
        self._history = None
        self.n_etg += cnt.n_etg
        self.n_attempts += cnt.n_attempts
        self.n_decoh += cnt.n_decoh
        return self

    def save_etg(self, task: "EntangleTask") -> None:
        self.n_etg += 1
        self.n_attempts += task.k
        if self._history is not None:
            self._history.append(task.t_finish.slot - task.t_reserve.slot)

    @property
    def etg_rate(self) -> float:
        """
        Entanglement rate, average number of entanglements per second per memory pair.

        This reflects the uncontested rate, as if the memory pair is always available for the LinkLayer
        to perform entanglement generations, not locked by the forwarding function.

        This property is available only if constructor received positive ``history`` and ``rtt``.
        If history is disabled or empty, returns NaN.
        """
        if not self._history:
            return np.nan
        avg_duration = self._rtt.slot + np.mean(self._history).item()
        if avg_duration <= 0:
            return np.nan
        return self._rtt.accuracy / avg_duration

    @property
    def decoh_ratio(self) -> float:
        """Decoherence ratio, ``n_decoh/n_etg``."""
        return self.n_decoh / self.n_etg if self.n_etg > 0 else 0

    def __repr__(self) -> str:
        return f"etg={self.n_etg} attempts={self.n_attempts} decoh={self.n_decoh} decoh_ratio={self.decoh_ratio}"
