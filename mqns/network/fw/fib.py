#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
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

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Final, Literal, final

import numpy as np

from mqns.network.fw.message import SwapSequence
from mqns.simulator import Time, func_to_event

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class PathDirection(Enum):
    """
    Direction left or right.
    """

    L = 0
    """
    Leftward.
    """
    R = 1
    """
    Rightward.
    """


@final
class FibPath:
    """FIB path entry --- information related to a path."""

    __slots__ = ("req", "path_id", "route", "own_idx", "swap", "swap_cutoff", "purif", "sg")

    req: "FibRequest"
    """Request entry reference, assigned by ``FibRequest.__init__``."""
    path_id: Final[int]
    """Path identifier, identifies end-to-end path."""

    route: Final[Sequence[str]]
    """List of nodes traversed by the path."""
    own_idx: Final[int]
    """Index of own node within the route."""
    swap: Final[SwapSequence]
    """Swap sequence."""
    swap_cutoff: Final[Sequence[Time | None]]
    """Swap cutoff times."""
    purif: Final[Mapping[str, int]]
    """Purification scheme."""
    sg: Final["FibSwapGroup|None"]
    """Swap group that this node belongs to; None for end-node or swap-disabled."""

    def __init__(
        self,
        *,
        path_id: int,
        route: Sequence[str],
        own_idx: int,
        swap: SwapSequence,
        swap_cutoff: Sequence[Time | None],
        purif: Mapping[str, int],
    ):
        self.path_id = path_id
        self.route = route
        self.own_idx = own_idx
        self.swap = swap
        self.swap_cutoff = swap_cutoff
        self.purif = purif
        self.sg = None if self.own_swap_rank == self.swap[0] else FibSwapGroup(self)

    @property
    def req_id(self) -> int:
        """Request identifier."""
        return self.req.req_id

    @property
    def own_is_end_node(self) -> bool:
        """Whether own node is an end node."""
        return self.sg is None

    @property
    def own_swap_rank(self) -> int:
        """Swap rank of own node."""
        return self.swap[self.own_idx]


@final
class FibSwapGroup:
    """
    FibSwapGroup provides topological information to determine the heralding directions within
    a FIB route that may contain parallel swapping instructions.

    Each node within a FIB route, excluding the first and last end nodes, belongs to a swap group.
    The swap group of node *N* is identified with these steps:

    1. From ``entry.route``, discard nodes with a lower rank than node *N*.
    2. Find the longest continuous segment of nodes that contains *N*, where every node has the same rank as *N*.

    The left/right neighbor of a swap group is a node that has a higher rank and not part of the swap group.
    The leftmost/rightmost node of the swap group is responsible for heralding the left/right neighbor,
    if that neighbor would not be heralded by the opposite neighbor.
    """

    __slots__ = ("path_id", "rank", "nodes", "l_neigh", "r_neigh", "dir", "own_idx")

    path_id: Final[int]
    """FIB entry path ID."""

    rank: Final[int]
    """Rank of all nodes in this swap group."""

    nodes: Final[Sequence[str]]
    """Nodes in this swap group."""

    l_neigh: Final[str]
    """Left neighbor, not part of this swap group."""

    r_neigh: Final[str]
    """Right neighbor, not part of this swap group."""

    dir: Final[Literal["l", "b", "r"]]
    """
    Heralding direction when there is no swap failure.

    * b: Bidirectional -- Both neighbors have the same rank, or the segment requires purification.
    * l: Leftward -- Left neighbor has lower rank than right neighbor.
    * r: Rightward -- Right neighbor has lower rank than left neighbor.
    """

    own_idx: Final[int]
    """Index of own node within ``nodes``."""

    def __init__(self, fp: FibPath):
        route_len = len(fp.route)

        self.path_id = fp.path_id
        self.rank = fp.own_swap_rank

        nodes = [fp.route[fp.own_idx]]
        self.l_neigh, l_rank = self._extend_1d(fp, nodes, range(fp.own_idx - 1, -1, -1))
        self.own_idx = len(nodes) - 1
        nodes.reverse()
        self.r_neigh, r_rank = self._extend_1d(fp, nodes, range(fp.own_idx + 1, route_len))
        self.nodes = nodes

        if l_rank == r_rank or fp.purif.get(f"{self.l_neigh}-{self.r_neigh}", 0) > 0:
            self.dir = "b"
        elif l_rank < r_rank:
            self.dir = "l"
        else:
            self.dir = "r"

    def _extend_1d(self, entry: FibPath, nodes: list[str], index_range: Iterable[int]) -> tuple[str, int]:
        for i in index_range:
            node_rank = entry.swap[i]
            if node_rank > self.rank:
                return entry.route[i], entry.swap[i]
            if node_rank == self.rank:
                nodes.append(entry.route[i])
        raise ValueError(f"FibSwapGroup cannot find boundary in {entry.swap} from node index {entry.own_idx}")

    @property
    def l_most(self) -> bool:
        """Determine if own node is the leftmost node in the group."""
        return self.own_idx == 0

    @property
    def r_most(self) -> bool:
        """Determine if own node is the rightmost node in the group."""
        return self.own_idx == len(self.nodes) - 1

    @property
    def own_node(self) -> str:
        """Own node name."""
        return self.nodes[self.own_idx]

    @property
    def l_adj(self) -> str:
        """
        Left adjacent node, either same-ranked peer or higher-ranked neighbor.

        Own node should herald this node if allowed by ``dir`` or when there is a failure.
        """
        return self.l_neigh if self.l_most else self.nodes[self.own_idx - 1]

    @property
    def r_adj(self) -> str:
        """
        Right adjacent node, either same-ranked peer or higher-ranked neighbor.

        Own node should herald this node if allowed by ``dir`` or when there is a failure.
        """
        return self.r_neigh if self.r_most else self.nodes[self.own_idx + 1]

    def repr_route(self) -> str:
        return "".join(
            [
                self.l_neigh,
                "<" if self.dir in ("l", "b") else "~",
                "-".join(f"[{n}]" if i == self.own_idx else n for (i, n) in enumerate(self.nodes)),
                ">" if self.dir in ("b", "r") else "~",
                self.r_neigh,
            ]
        )

    def __repr__(self) -> str:
        return f"FibSwapGroup({self.repr_route()}, rank={self.rank})"


class FibRequest:
    """FIB request entry --- information related to an end-to-end request."""

    __slots__ = ("req_id", "src", "dst", "paths", "active_until", "epr_count_remain")

    req_id: Final[int]
    """Request identifier."""
    src: Final[str]
    """Source node."""
    dst: Final[str]
    """Destination node."""
    paths: Final[Sequence[FibPath]]
    """List of ``FibPath`` entries."""

    active_until: Time
    """
    Active period upper bound (exclusive), ``Time.MAX`` means no restriction.
    """
    epr_count_remain: int | float
    """
    If ``epr_count`` is restricted, remaining quantity of entangled pairs.
    Otherwise, positive infinity.
    This field is only used on end nodes.
    """

    def __init__(self, req_id: int, entries: Sequence[FibPath], *, epr_count=-1):
        self.req_id = req_id
        self.src, *_, self.dst = entries[0].route
        self.paths = entries
        self.active_until = Time.MAX
        self.epr_count_remain = np.inf if epr_count < 0 else epr_count

        for entry in entries:
            entry.req = self
            assert entry.route[0] == self.src
            assert entry.route[-1] == self.dst

    def is_active(self, now: Time) -> bool:
        """
        Determine if the request is active, subject to these conditions:

        * ``epr_count`` has not been reached.
        * Time point ``now`` is within active period.
        """
        return now < self.active_until and self.epr_count_remain > 0


class Fib:
    """
    Forwarding Information Base of a forwarder.
    """

    def __init__(self):
        self._reqs: dict[int, FibRequest] = {}  # keyed by req_id
        self._end_reqs = defaultdict[str, list[FibRequest]](list)  # own node is end-node, keyed by opposite end-node
        self._paths: dict[int, FibPath] = {}  # keyed by path_id

    def install(self, fw: "Forwarder") -> None:
        self.simulator = fw.simulator
        self._own_name = fw.node.name

    def clear(self) -> None:
        """
        Clear all tables.
        """
        self._reqs.clear()
        self._end_reqs.clear()
        self._paths.clear()

    def get_path(self, path_id: int) -> FibPath:
        """
        Retrieve FIB path entry by path identifier.

        Raises:
            LookupError: Path entry not found.
        """
        try:
            return self._paths[path_id]
        except KeyError:
            raise LookupError(f"FibPath({path_id}) not found") from None

    def get_req(self, req_id: int) -> FibRequest:
        """
        Retrieve FIB request entry by request identifier.

        Raises:
            LookupError: Request entry not found.
        """
        try:
            return self._reqs[req_id]
        except KeyError:
            raise LookupError(f"FibRequest({req_id}) not found") from None

    def _req_opposite_end(self, fr: FibRequest) -> str | None:
        if fr.src == self._own_name:
            return fr.dst
        if fr.dst == self._own_name:
            return fr.src
        return None

    def insert_req(self, fr: FibRequest) -> None:
        """
        Insert a request entry and all its path entries.
        """
        if fr.req_id in self._reqs:
            raise ValueError(f"FibRequest({fr.req_id}) already exists")
        self._reqs[fr.req_id] = fr

        for entry in fr.paths:
            if entry.path_id in self._paths:
                raise ValueError(f"FibPath({entry.path_id}) already exists")
            self._paths[entry.path_id] = entry

        if opposite := self._req_opposite_end(fr):
            self._end_reqs[opposite].append(fr)

    def delete_req(self, req_id: int, *, delay: Time) -> FibRequest:
        """
        Schedule the deletion of a request entry and all its path entries.

        Args:
            delay: Delay for the final deletion.

        Returns:
            The FIB request entry.
        """
        fr = self.get_req(req_id)
        fr.active_until = self.simulator.tc
        self.simulator.sched(func_to_event(fr.active_until + delay, self._delete_req, req_id))
        return fr

    def _delete_req(self, req_id: int) -> None:
        fr = self.get_req(req_id)
        del self._reqs[req_id]

        for entry in fr.paths:
            del self._paths[entry.path_id]

        if opposite := self._req_opposite_end(fr):
            l = self._end_reqs[opposite]
            l.remove(fr)
            if not l:
                del self._end_reqs[opposite]

    def list_end_reqs(self, opposite_end: str) -> list[FibRequest]:
        """
        List FIB request entries where own node is an end-node.

        Args:
            opposite_end: The opposite end-node.
        """
        return self._end_reqs.get(opposite_end, [])
