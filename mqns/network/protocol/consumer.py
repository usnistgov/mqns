from collections import defaultdict
from typing import Protocol, overload, override

import numpy as np

from mqns.entity.memory import QubitState
from mqns.entity.node import Application, NodePair, QNode, split_node_pair
from mqns.network.ll import QubitConsumeEvent, QubitReleasedEvent
from mqns.network.network import QuantumNetwork
from mqns.simulator import Time, event_handler
from mqns.utils import json_encodable


class RequestLike(Protocol):
    @property
    def req_id(self) -> int: ...
    @property
    def src(self) -> str: ...
    @property
    def dst(self) -> str: ...


@json_encodable
class RequestCounters:
    """
    Consumption counters for an end-to-end request.

    Each entangled pair is delivered to two consumers, possibly at different times due to heralding timing.
    However, only the consumer that measures the second qubit logs the consumption,
    because the final end-to-end fidelity can only be calculated after applying memory error models on both sides.
    """

    n_consumed = 0
    """How many entanglements were consumed (either end-to-end or in swap-disabled mode)."""
    consumed_sum_fidelity = 0.0
    """
    Sum of fidelity of consumed entanglements.
    """
    consumed_fidelity_values: list[float] | None = None
    """
    Fidelity values of consumed entanglements, None disables collection.
    """
    t_first = Time.MAX
    """
    Timestamp of first consumption.
    """
    t_last = Time.MIN
    """
    Timestamp of last consumption.
    """

    @staticmethod
    def _parse_req(arg1: RequestLike | int, np: NodePair) -> tuple[int, str, str]:
        if isinstance(arg1, int):
            return arg1, *split_node_pair(np)
        return arg1.req_id, arg1.src, arg1.dst

    @staticmethod
    @overload
    def of(net: QuantumNetwork, req: RequestLike, /) -> "RequestCounters":
        """
        Obtain consumption counters of a request.

        Args:
            net: Quantum network.
            req: ``Request`` or ``RoutingPath`` instance.

        Returns: ConsumerPathCounters for the request, aggregated from src and dst nodes.
        """

    @staticmethod
    @overload
    def of(net: QuantumNetwork, req_id: int, np: NodePair, /) -> "RequestCounters":
        """
        Obtain consumption counters of a request.

        Args:
            net: Quantum network.
            req_id: Request ID.
            np: Two node names.

        Returns: ConsumerPathCounters for the request, aggregated from src and dst nodes.
        """

    @staticmethod
    def of(net: QuantumNetwork, arg1: RequestLike | int, np: NodePair = "") -> "RequestCounters":
        req_id, src, dst = RequestCounters._parse_req(arg1, np)
        a = net.get_node(src).get_app(Consumer).cnt.get(req_id, RequestCounters())
        b = net.get_node(dst).get_app(Consumer).cnt.get(req_id, RequestCounters())

        g = RequestCounters()
        g.n_consumed = a.n_consumed + b.n_consumed
        g.consumed_sum_fidelity = a.consumed_sum_fidelity + b.consumed_sum_fidelity
        if a.consumed_fidelity_values is b.consumed_fidelity_values:
            g.consumed_fidelity_values = a.consumed_fidelity_values
        g.t_first = min(a.t_first, b.t_first)
        g.t_last = max(a.t_last, b.t_last)
        return g

    @staticmethod
    @overload
    def enable_collect_all(net: QuantumNetwork, req: RequestLike, /) -> None:
        """
        Enable collection of all fidelity values.

        Args:
            net: Quantum network.
            req: ``RoutingPath`` instance.
        """

    @staticmethod
    @overload
    def enable_collect_all(net: QuantumNetwork, req_id: int, np: NodePair, /) -> None:
        """
        Enable collection of all fidelity values.

        Args:
            net: Quantum network.
            req_id: Request ID.
            np: Two node names.
        """

    @staticmethod
    def enable_collect_all(net: QuantumNetwork, arg1: RequestLike | int, np: NodePair = "") -> None:
        req_id, src, dst = RequestCounters._parse_req(arg1, np)
        a = net.get_node(src).get_app(Consumer).cnt[req_id]
        b = net.get_node(dst).get_app(Consumer).cnt[req_id]

        assert a.consumed_fidelity_values is None
        assert b.consumed_fidelity_values is None
        a.consumed_fidelity_values = b.consumed_fidelity_values = []

    def save(self, t: Time, fidelity: float) -> None:
        if self.n_consumed == 0:
            self.t_first = t
        self.t_last = t
        self.n_consumed += 1
        self.consumed_sum_fidelity += fidelity
        if self.consumed_fidelity_values is not None:
            self.consumed_fidelity_values.append(fidelity)

    @property
    def consumed_avg_fidelity(self) -> float:
        """Average fidelity of consumed entanglements."""
        if self.consumed_fidelity_values is not None and len(self.consumed_fidelity_values) == self.n_consumed > 0:
            return np.mean(self.consumed_fidelity_values).item()
        return self.get_per_consumed(self.consumed_sum_fidelity)

    def get_rate(self, duration: float) -> float:
        """
        Calculate entanglement rate.

        Args:
            duration: How many seconds did the path remain active.

        Returns: Entanglement rate in entanglements per second.
        """
        return self.n_consumed / duration

    def get_latency(self, active_since: Time) -> tuple[float, float]:
        """
        Calculate satisfaction latency.

        Args:
            active_since: When did the request become active.

        Returns: Latency of first and last EPR; nan if ``n_consumed`` is zero.
        """
        if self.n_consumed == 0:
            return np.nan, np.nan
        return (self.t_first - active_since).sec, (self.t_last - active_since).sec

    def get_per_consumed(self, x: float) -> float:
        """
        Divide a value by ``n_consumed``, but return zero if ``n_consumed`` is zero.
        """
        return x / self.n_consumed if self.n_consumed > 0 else 0.0

    def __repr__(self) -> str:
        return f"consumed={self.n_consumed} (F={self.consumed_avg_fidelity})"


class Consumer(Application[QNode]):
    """
    T-class application that consumes entanglements.
    """

    def __init__(self):
        self.cnt = defaultdict[int, RequestCounters](RequestCounters)
        """
        Path counters keyed by req_id.
        """

    @override
    def install(self, node):
        self._application_install(node, QNode)
        self.memory = self.node.memory
        """Quantum memory of the node."""

    @event_handler
    def handle_consume(self, event: QubitConsumeEvent) -> None:
        now = event.t
        qubit = event.qubit
        epr = event.epr
        req_id = event.req_id

        role_str = "first"
        if epr.consume_with_store_decay_side(now, side=0 if epr.src is self.node else 1):
            role_str = "second"
            self.cnt[req_id].save(now, epr.fidelity)
        self.log_debug("consume EPR %s for request %s: %s", role_str, req_id, epr)

        self.memory.read(qubit.addr, remove=True)
        qubit.state = QubitState.RELEASE
        self.simulator.sched(QubitReleasedEvent(self.node, qubit, t=now))
