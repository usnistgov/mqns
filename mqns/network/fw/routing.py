import itertools
from collections.abc import Mapping, Sequence
from typing import Final, Literal, Protocol, TypedDict, Unpack

from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.fw.message import MultiplexingVector, PathInstructions
from mqns.network.fw.swap_sequence import SwapSequenceInput, parse_swap_sequence
from mqns.network.route import RouteQueryResult
from mqns.simulator import Time

type MultipathOption = Literal["all", "any"]
"""
How to use multiple static or computed routes, defaults to "all".

* "all": All routes are installed, each under a separate ``path_id``.
         Each route can complete swaps and deliver entanglements for the request.
         If there are insufficient network resources for all of the routes,
         the request would be deferred or rejected.
* "any": Only one route is installed, depending on available network resources.
         If there are insufficient network resources for any of the routes,
         the request would be deferred or rejected.
"""

type MultiplexingVectorInput = Literal["max"] | int | MultiplexingVector
"""
Buffer-space multiplexing vector or how to generate them.

* "max": Allocate the maximum quantity of qubits per quantum channel, depending on channel capacity.
  * If paths from multiple ``RoutingPath`` objects shares the same channel,
    this would likely cause a conflict.
  * If a ``RoutingPath`` generates multiple paths (typically when used with ``YenRouteAlgorithm``)
    where the same channel is shared by several paths, the channel capacity is equally divided among them.
* Zero: Allocate the maximum quantity of qubits per quantum channel, depending on channel capacity.
  * If multiple paths from one or more ``RoutingPath`` objects share the same channel,
    this would likely cause a conflict.
* Positive integer: Allocate specific number of qubits per quantum channel.
* ``MultiplexingVector``: Use the pre-defined multiplexing vector, which must match route length.
"""


class RoutingPathInitArgs(TypedDict, total=False):
    req_id: int
    """Request identifier, defaults to auto-assignment."""
    multipath: MultipathOption
    """
    How to use multiple routes, defaults to "all".
    """
    bufferspace_mv: MultiplexingVectorInput
    """
    Buffer-space multiplexing vector or how to generate them, defaults to "max".
    This field has no effect if the network is not using buffer-space multiplexing scheme.
    """
    swap: SwapSequenceInput
    """Swap sequence or swap policy, defaults to ASAP."""
    swap_cutoff: Sequence[float] | None
    """Swap cut-off times in seconds."""
    purif: Mapping[str, int] | None
    """Purification scheme."""


class ComputeRoutesContext(Protocol):
    """
    Functions provided by ``RoutingController`` for use by ``RoutingPath.compute_routes()``.
    """

    @property
    def time_accuracy(self) -> int: ...

    def get_qchannel(self, a: str, b: str, /) -> QuantumChannel: ...

    def query_route(self, src: str, dst: str, /) -> Sequence[RouteQueryResult[QNode]]: ...

    def choose_ll_dir(self, a: str, b: str, /) -> Literal["R", "L"]:
        """
        Determine LinkLayer direction of a channel between ``a`` and ``b``.

        Returns:
            "R" makes ``a`` primary; "L" makes ``b`` primary.
        """
        ...


class RoutingPath:
    """
    ``RoutingPath`` computes ``PathInstructions`` from ``mqns.network.network.Request``.

    It stores attributes related to path computation.
    ``RoutingController`` calls the ``RoutingPath.compute_routes()`` method that returns one or more
    ``PathInstructions``, which are then sent to the forwarders.
    """

    src: Final[str]
    """
    End node name at the source (left) side.
    """

    dst: Final[str]
    """
    End node name at the destination (right) side.
    """

    req_id: int
    """
    Request identifier.
    """

    static_paths: list[list[str]] | None = None
    """
    Static paths.
    If non-empty, routing algorithm is not queried.
    """

    multipath: Final[MultipathOption]
    """
    How to use multiple routes, defaults to "all".
    """

    bufferspace_mv: Final[MultiplexingVectorInput]
    """
    Buffer-space multiplexing vector or how to generate them.
    This field has no effect if the network is not using buffer-space multiplexing scheme.
    """

    swap: Final[SwapSequenceInput]
    """
    Swap sequence or swap policy.
    """

    swap_cutoff: Final[Sequence[float] | None]
    """
    Swap cut-off values in seconds.
    """

    purif: Final[dict[str, int]]
    """
    Purification scheme.
    """

    def __init__(self, src: str, dst: str, /, **kwargs: Unpack[RoutingPathInitArgs]):
        """
        Constructor.

        Args:
            src: End node name at the source (left) side.
            dst: End node name at the destination (right) side.
        """
        self.src = src
        self.dst = dst
        self.req_id = kwargs.get("req_id", -1)
        self.multipath = kwargs.get("multipath", "all")
        self.bufferspace_mv = kwargs.get("bufferspace_mv", "max")
        self.swap = kwargs.get("swap") or "asap"
        self.swap_cutoff = kwargs.get("swap_cutoff")
        self.purif = dict(kwargs.get("purif") or {})

    @staticmethod
    def static(*paths: Sequence[str], **kwargs: Unpack[RoutingPathInitArgs]) -> "RoutingPath":
        """
        Construct ``RoutingPath`` from one or more static paths.

        ``src`` and ``dst`` are automatically extracted from the given paths.
        All paths must have the same two end nodes.
        """
        rp = RoutingPath(paths[0][0], paths[0][-1], **kwargs)
        rp.static_paths = []
        for p in paths:
            path = list(p)
            assert path[0] == rp.src
            assert path[-1] == rp.dst
            rp.static_paths.append(path)
        return rp

    def compute_paths(self, ctx: ComputeRoutesContext) -> list[PathInstructions]:
        """
        Compute and return a list of path instructions.

        Pre-conditions:

        * ``self.req_id`` is assigned to a non-negative value.

        Returns:
            List of path instructions with ``route``, ``ll_dir``, ``swap``, ``swap_cutoff``, ``purif`` fields filled.
            The ``RoutingController`` subclass must overwrite ``path_id`` and populate other fields.
        """
        assert self.req_id >= 0

        # Compute shortest paths.
        if self.static_paths:
            paths = self.static_paths
        else:
            routes = ctx.query_route(self.src, self.dst)
            paths = [route.path for route in routes]

        insts: list[PathInstructions] = []
        for path in paths:
            inst = PathInstructions(
                path_id=-1,
                route=list(path),
                ll_dir="".join(ctx.choose_ll_dir(a, b) for a, b in itertools.pairwise(path)),
                swap=parse_swap_sequence(self.swap, path),
                purif=self.purif,
            )

            if self.swap_cutoff:
                inst["swap_cutoff"] = [-1 if t < 0 else Time.sec_to_slot(t, ctx.time_accuracy) for t in self.swap_cutoff]

            insts.append(inst)

        return insts
