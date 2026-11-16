import itertools
from collections.abc import Callable, Iterable, Mapping, Set
from typing import TYPE_CHECKING, Final, final

from mqns.network.fw import PathInstructions
from mqns.network.proactive.fw_nb import to_fib_erase_delay
from mqns.simulator import Simulator, Time, func_to_event
from mqns.utils import unwrap

if TYPE_CHECKING:
    from mqns.network.proactive.controller import NodeResourceInfo


class NodeUtilization:
    """Controller's view of node resource utilization."""

    name: Final[str]
    """Node name."""

    ch_cap: Final[Mapping[str, int]]
    """
    Per-channel configured capacity.
    Only used for resolving ``mv="max"`` and ``mv=0``.
    """

    t_cohere: Final[Time]
    """Memory coherence time."""

    n_qubits: Final[int]
    """How many memory qubits exist in this node."""

    avail_qubits: int
    """How many memory qubits are available for allocation."""

    def __init__(self, simulator: Simulator, node: str, nri: "NodeResourceInfo"):
        self.name = node
        self.ch_cap = nri.ch_cap
        self.t_cohere = simulator.time(sec=nri.t_cohere)
        self.n_qubits = nri.n_qubits
        self.avail_qubits = self.n_qubits

    def __repr__(self) -> str:
        return f"{self.name}({self.avail_qubits}/{self.n_qubits})"


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

        Call :py:class:`PathDemands.commit` to commit the resources.
        """
        demands = PathDemands(self)
        for inst in insts:
            demands.add_path(inst)
        return demands


@final
class PathDemands:
    """
    Track the resource demands for one or more paths.

    This is created by :py:meth:`ResourceUtilization.gather_demands`.
    """

    violations: Set[str] = frozenset(["#UNKNOWN"])

    def __init__(self, ru: ResourceUtilization):
        self.ru: Final = ru
        self.demands: Final[dict[NodeUtilization, int]] = {}
        self.max_t_cohere = Time.MIN

    def __repr__(self) -> str:
        return f"PathDemands({[f'{ncu.name}:{n}' for ncu, n in self.demands.items()]}, violations={sorted(self.violations)})"

    def add_path(self, inst: PathInstructions) -> None:
        """
        Add a path to the resource demands.
        """
        mv = unwrap(inst.get("bufferspace_mv"))
        for i, (a, b) in enumerate(itertools.pairwise(inst["route"])):
            for j, node in (0, a), (1, b):
                self._add_node(node, mv[2 * i + j])

    def _add_node(self, node: str, n: int) -> None:
        nu = self.ru.nodes[node]
        self.demands[nu] = self.demands.get(nu, 0) + n

    def commit(self) -> bool:
        """
        Commit the resources for these paths, if possible.

        Returns:
            True if there are sufficient resources and they have been committed.
            False if there are resource violations, which may be retrieved via :py:attr:`violations`.
        """
        self.violations = {nu.name for nu, demand in self.demands.items() if demand > nu.avail_qubits}
        if len(self.violations) > 0:
            return False

        for nu, demand in self.demands.items():
            nu.avail_qubits -= demand
            self.max_t_cohere = max(self.max_t_cohere, nu.t_cohere)
        return True

    def release(self, *, cb_after: Callable[[], None]) -> None:
        """
        Release the resources for these paths.

        The release is internally delayed by a duration equal to forwarder's FIB erase delay.

        Pre-conditions:

        * The resources was committed, i.e. :py:meth:`commit` returned True.
        """
        assert self.max_t_cohere.accuracy > 0, "PathDemands was not committed"
        simulator = self.ru.simulator
        fib_erase_delay = to_fib_erase_delay(self.max_t_cohere)
        simulator.sched(func_to_event(simulator.tc + fib_erase_delay, self._revert, cb_after))

    def _revert(self, cb_after: Callable[[], None]) -> None:
        for nu, demand in self.demands.items():
            nu.avail_qubits += demand
        cb_after()
