import itertools
from collections.abc import Callable, Iterable, Mapping, Set
from typing import TYPE_CHECKING, Final, final

from mqns.network.fw import PathInstructions
from mqns.network.proactive.fw_nb import to_fib_erase_delay
from mqns.simulator import Simulator, Time, func_to_event

if TYPE_CHECKING:
    from mqns.network.proactive.controller import NodeResourceInfo


class NodeChannelUtilization:
    """Controller's view of utilization of qubits assigned to a channel."""

    nu: Final["NodeUtilization"]
    """Node utilization reference."""

    neighbor: Final[str]
    """Neighbor name."""

    n_qubits: Final[int]
    """How many memory qubits are assigned to this channel."""

    unallocated_qubits: int
    """How many memory qubits are available for allocation to paths."""

    def __init__(self, nu: "NodeUtilization", neighbor: str, n_qubits: int):
        self.nu = nu
        self.neighbor = neighbor
        self.n_qubits = n_qubits
        self.unallocated_qubits = self.n_qubits

    def __repr__(self) -> str:
        return f"{self.neighbor}({self.unallocated_qubits}/{self.n_qubits})"

    @property
    def name(self) -> str:
        return f"{self.nu.name}-{self.neighbor}"


class NodeUtilization:
    """Controller's view of node resource utilization."""

    name: Final[str]
    """Node name."""

    channels: Final[dict[str, NodeChannelUtilization]]
    """Per-channel assignment."""

    t_cohere: Final[Time]
    """Memory coherence time."""

    n_qubits: Final[int]
    """How many memory qubits exist in this node."""

    unassigned_qubits: int
    """How many memory qubits are available for assignment to channels."""

    def __init__(self, simulator: Simulator, node: str, nri: "NodeResourceInfo"):
        self.name = node
        self.t_cohere = simulator.time(sec=nri.t_cohere)
        self.n_qubits = nri.n_qubits
        self.unassigned_qubits = self.n_qubits

        self.channels = {}
        for neighbor, n in nri.ch_cap.items():
            ncu = NodeChannelUtilization(self, neighbor, n)
            self.channels[ncu.neighbor] = ncu
            self.unassigned_qubits -= ncu.n_qubits

    def __repr__(self) -> str:
        return (
            f"{self.name}({self.unassigned_qubits}/{self.n_qubits}; "
            f"channels={[ncu.__repr__() for ncu in self.channels.values()]})"
        )


class ResourceUtilization:
    """Controller's view of network-wide resource utilization."""

    simulator: Final[Simulator]

    nodes: Final[Mapping[str, NodeUtilization]]
    """Node utilization keyed by node name."""

    def __init__(self, simulator: Simulator, nri: Mapping[str, "NodeResourceInfo"]):
        self.simulator = simulator
        self.nodes = {k: NodeUtilization(simulator, k, v) for k, v in nri.items()}

    def __repr__(self) -> str:
        return f"ResourceUtilization(nodes={[nu.__repr__() for nu in self.nodes.values()]})"

    def gather_demands(self, insts: Iterable[PathInstructions]) -> "PathDemands":
        """
        Compute resource demands for one or more paths.

        If the returned ``PathDemands`` indicates no violations, call ``demands.commit()`` to commit the resources.
        """
        demands = PathDemands(self)
        for inst in insts:
            demands.add_path(inst)
        return demands


@final
class PathDemands:
    """
    Track the resource demands for one or more paths.
    """

    violations: Set[str] = frozenset(["#UNKNOWN"])

    def __init__(self, ru: ResourceUtilization):
        self.ru: Final = ru
        self.demands: Final[dict[NodeChannelUtilization, int]] = {}
        self.max_t_cohere = Time.MIN

    def __repr__(self) -> str:
        return f"PathDemands({[f'{ncu.name}:{n}' for ncu, n in self.demands.items()]}, violations={sorted(self.violations)})"

    def add_path(self, inst: PathInstructions) -> None:
        """
        Add a path to the resource demands.
        """
        assert "bufferspace_mv" in inst
        mv = inst["bufferspace_mv"]
        for i, (a, b) in enumerate(itertools.pairwise(inst["route"])):
            for j, node, neighbor in (0, a, b), (1, b, a):
                self._add_node(node, neighbor, mv[2 * i + j])

    def _add_node(self, node: str, neighbor: str, n: int) -> None:
        ncu = self.ru.nodes[node].channels[neighbor]
        self.demands[ncu] = self.demands.get(ncu, 0) + n

    def find_violations(self) -> bool:
        """
        Determine whether committing these paths would cause resource violations.
        """
        return len(self.violations) > 0

    def commit(self) -> bool:
        """
        Commit the resources for these paths, if possible.

        Returns:
            True if there are sufficient resources and they have been committed.
            False if there are resource violations, which may be retrieved via ``.violations``.
        """
        self.violations = {ncu.name for ncu, demand in self.demands.items() if demand > ncu.unallocated_qubits}
        if len(self.violations) > 0:
            return False

        for ncu, demand in self.demands.items():
            ncu.unallocated_qubits -= demand
            self.max_t_cohere = max(self.max_t_cohere, ncu.nu.t_cohere)
        return True

    def release(self, *, cb_after: Callable[[], None]) -> None:
        """
        Release the resources for these paths.

        The release is internally delayed by a duration equal to forwarder's FIB erase delay.

        Pre-conditions:

        * The resources was committed, i.e. ``commit()`` returned True.
        """
        assert self.max_t_cohere is not Time.MIN, "PathDemands was not committed"
        simulator = self.ru.simulator
        fib_erase_delay = to_fib_erase_delay(self.max_t_cohere)
        simulator.sched(func_to_event(simulator.tc + fib_erase_delay, self._revert, cb_after))

    def _revert(self, cb_after: Callable[[], None]) -> None:
        for ncu, demand in self.demands.items():
            ncu.unallocated_qubits += demand
        cb_after()
