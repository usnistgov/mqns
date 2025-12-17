from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from enum import Enum, auto
from itertools import pairwise

from typing_extensions import override

from mqns.network.network import QuantumNetwork
from mqns.network.proactive.message import MultiplexingVector, PathInstructions, make_path_instructions
from mqns.network.proactive.swap_sequence import parse_swap_sequence
from mqns.simulator import Simulator, Time
from mqns.utils import log


class QubitAllocationType(Enum):
    DISABLED = auto()
    """Disable multiplexing vector, for use with statistical or dynamic EPR multiplexing schemes."""
    MIN_CAPACITY = auto()
    """Compute buffer-space multiplexing vector based on minimum memory capacity."""
    FOLLOW_QCHANNEL = auto()
    """Compute buffer-space multiplexing vector based on qubit-qchannel assignments."""


def _compute_mv(net: QuantumNetwork, route: list[str], qubit_allocation: QubitAllocationType) -> MultiplexingVector | None:
    """
    Compute buffer-space multiplexing vector.
    """
    match qubit_allocation:
        case QubitAllocationType.DISABLED:
            return None
        case QubitAllocationType.MIN_CAPACITY:
            c = [net.get_node(node_name).memory.capacity for node_name in route]
            c[0] *= 2
            c[-1] *= 2
            q = min(c) // 2
            return [(q, q) for _ in range(len(route) - 1)]
        case QubitAllocationType.FOLLOW_QCHANNEL:
            return [(0, 0) for _ in range(len(route) - 1)]
        case _:
            raise ValueError("unknown qubit_allocation")


def _parse_swap_cutoff(simulator: Simulator, input: list[float] | None) -> list[Time | None] | None:
    if input is None:
        return None
    return [simulator.time(sec=t) if t >= 0 else None for t in input]


class RoutingPath(ABC):
    """
    Compute routing path(s) for installing through ProactiveRoutingController.
    """

    def __init__(self, src: str, dst: str, req_id: int | None, path_id: int | None):
        self.src = src
        """
        Source node name.
        """
        self.dst = dst
        """
        Destination node name.
        """
        self.req_id = -1 if req_id is None else req_id
        """
        Request identifier.

        If unspecified, the controller will assign the next unused value before calling `compute_paths`.
        """
        self.path_id = -1 if path_id is None else path_id
        """
        Path identifier for the first path.

        If unspecified, the controller will assign the next unused value before calling `compute_paths`.

        When `compute_paths` yields multiple paths, this is the path_id on the first path,
        while subsequent paths are given consecutive path_ids.
        """

    @abstractmethod
    def compute_paths(self, net: QuantumNetwork) -> Iterator[PathInstructions]:
        """
        Compute and yield one or more path instructions.

        Args:
            net: The quantum network.
                 `net.build_route()` has been called prior to invoking this function.

        Returns:
            A generator of path instructions.
            They will be installed into the nodes.
        """
        pass

    def query_routes(self, net: QuantumNetwork) -> Iterator[list[str]]:
        """
        Query routes from source node to destination node.
        """
        src = net.get_node(self.src)
        dst = net.get_node(self.dst)
        route_result = net.query_route(src, dst)
        if len(route_result) == 0:
            raise RuntimeError(f"ROUTING: No route from {src} to {dst}")
        for _, _, route_nodes in route_result:
            yield [node.name for node in route_nodes]


class RoutingPathStatic(RoutingPath):
    """
    Define a static routing path for installing through ProactiveRoutingController.
    """

    def __init__(
        self,
        route: list[str],
        *,
        req_id: int | None = None,
        path_id: int | None = None,
        swap: list[int] | str,
        swap_cutoff: list[float] | None = None,
        m_v: MultiplexingVector | QubitAllocationType = QubitAllocationType.FOLLOW_QCHANNEL,
        purif: dict[str, int] = {},
    ):
        super().__init__(route[0], route[-1], req_id, path_id)
        self.route = route
        self.swap = swap
        self.swap_cutoff = swap_cutoff
        self.m_v = m_v
        self.purif = purif

    @override
    def compute_paths(self, net: QuantumNetwork) -> Iterator[PathInstructions]:
        yield make_path_instructions(
            self.req_id,
            self.route,
            parse_swap_sequence(self.swap, self.route),
            _parse_swap_cutoff(net.simulator, self.swap_cutoff),
            _compute_mv(net, self.route, self.m_v) if isinstance(self.m_v, QubitAllocationType) else self.m_v,
            self.purif,
        )


class RoutingPathSingle(RoutingPath):
    """
    Compute a single shortest path for installing through ProactiveRoutingController.
    """

    def __init__(
        self,
        src: str,
        dst: str,
        *,
        req_id: int | None = None,
        path_id: int | None = None,
        qubit_allocation=QubitAllocationType.FOLLOW_QCHANNEL,
        swap: list[int] | str,
        swap_cutoff: list[float] | None = None,
        purif: dict[str, int] = {},
    ):
        super().__init__(src, dst, req_id, path_id)
        self.qubit_allocation = qubit_allocation
        self.swap = swap
        self.swap_cutoff = swap_cutoff
        self.purif = purif

    @override
    def compute_paths(self, net: QuantumNetwork) -> Iterator[PathInstructions]:
        route = next(self.query_routes(net))
        log.debug(f"ROUTING: Computed path #{self.path_id}: {route}")
        yield make_path_instructions(
            self.req_id,
            route,
            parse_swap_sequence(self.swap, route),
            _parse_swap_cutoff(net.simulator, self.swap_cutoff),
            _compute_mv(net, route, self.qubit_allocation),
            self.purif,
        )


class RoutingPathMulti(RoutingPath):
    """
    Compute multiple shortest paths for installing through ProactiveRoutingController.

    This should be used with YenRouteAlgorithm in the QuantumNetwork.
    The number of paths for each request is determined by the routing algorithm.

    This is only compatible with buffer-space multiplexing scheme.
    """

    def __init__(
        self,
        src: str,
        dst: str,
        *,
        req_id: int | None = None,
        path_id: int | None = None,
        swap: list[int] | str,
        swap_cutoff: list[float] | None = None,
        purif: dict[str, int] = {},
    ):
        super().__init__(src, dst, req_id, path_id)
        self.swap = swap
        self.swap_cutoff = swap_cutoff
        self.purif = purif

    @override
    def compute_paths(self, net: QuantumNetwork) -> Iterator[PathInstructions]:
        # Get all shortest paths (M ≥ 1)
        routes = list(self.query_routes(net))

        # Count usage of each quantum channel across all paths
        qchannel_use_count = defaultdict[str, int](lambda: 0)
        for route in routes:
            for name_a, name_b in pairwise(route):
                ch = net.get_qchannel(name_a, name_b)
                qchannel_use_count[ch.name] += 1

        # Process each path
        for path_id_add, route in enumerate(routes):
            path_id = self.path_id + path_id_add
            log.debug(f"ROUTING: Computed path #{path_id}: {route}")

            # Compute buffer-space multiplexing vector as pairs of (qubits_at_node_i, qubits_at_node_i+1)
            # The qubits are divided among all paths that share the qchannel
            m_v: MultiplexingVector = []
            for name_a, name_b in pairwise(route):
                node_a = net.get_node(name_a)
                node_b = net.get_node(name_b)
                ch = net.get_qchannel(name_a, name_b)
                shared = qchannel_use_count.get(ch.name)
                assert shared is not None

                qubits_a = sum(1 for _ in node_a.memory.find(lambda *_: True, qchannel=ch))
                qubits_b = sum(1 for _ in node_b.memory.find(lambda *_: True, qchannel=ch))
                if shared > 0:
                    qubits_a //= shared
                    qubits_b //= shared

                m_v.append((qubits_a, qubits_b))

            # Send install instruction to each node on this path
            yield make_path_instructions(
                self.req_id,
                route,
                parse_swap_sequence(self.swap, route),
                _parse_swap_cutoff(net.simulator, self.swap_cutoff),
                m_v,
                self.purif,
            )
