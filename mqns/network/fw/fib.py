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

from collections.abc import Callable, Iterable, Iterator, Set
from dataclasses import dataclass
from typing import Literal, final

from mqns.network.fw.message import SwapSequence
from mqns.simulator import Time


@final
@dataclass(frozen=True)
class FibEntry:
    path_id: int
    """Path identifier, identifies end-to-end path."""
    req_id: int
    """Request identifier, identifies source-destination pair."""
    route: list[str]
    """List of nodes traversed by the path."""
    own_idx: int
    """Index of own node within the route."""
    swap: SwapSequence
    """Swap sequence."""
    swap_cutoff: list[Time | None]
    """Swap cutoff times."""
    purif: dict[str, int]
    """Purification scheme."""

    @property
    def own_swap_rank(self) -> int:
        return self.swap[self.own_idx]

    @property
    def is_swap_disabled(self) -> bool:
        """
        Determine whether swapping has been disabled.

        To disable swapping, set SwapSequence to a list of zeros.

        When swapping is disabled, the forwarder will consume entanglement upon completing purification,
        without attempting entanglement swapping.
        """
        return self.swap[0] == 0 == self.swap[-1]

    def find_index_and_swap_rank(self, node_name: str) -> tuple[int, int]:
        """
        Determine the swapping rank of a node.

        Args:
            node_name: a node name that exists in route.

        Returns:
            [0]: The node index in the route.
            [1]: Swapping rank of the node, explained in ``PathInstructions``.

        Raises:
            IndexError: node does not exist in route.
        """
        idx = self.route.index(node_name)
        return idx, self.swap[idx]


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

    rank: int
    """Rank of all nodes in this swap group."""

    nodes: list[str]
    """Nodes in this swap group."""

    l_neigh: str
    """Left neighbor, not part of this swap group."""

    r_neigh: str
    """Right neighbor, not part of this swap group."""

    dir: Literal["l", "b", "r"]
    """
    Heralding direction when there is no swap failure.

    * b: Bidirectional -- Both neighbors have the same rank, or the segment requires purification.
    * l: Leftward -- Left neighbor has lower rank than right neighbor.
    * r: Rightward -- Right neighbor has lower rank than left neighbor.
    """

    own_idx: int
    """Index of own node within ``nodes``."""

    l_adj: str
    """
    Left adjacent node, either same-ranked peer or higher-ranked neighbor.

    Own node should herald this node if allowed by ``dir`` or when there is a failure.
    """

    r_adj: str
    """
    Right adjacent node, either same-ranked peer or higher-ranked neighbor.

    Own node should herald this node if allowed by ``dir`` or when there is a failure.
    """

    @staticmethod
    def compute(entry: FibEntry) -> "FibSwapGroup":
        # TODO implement caching

        route_len = len(entry.route)
        if entry.own_idx in (0, route_len - 1):
            raise ValueError("FibSwapGroup is undefined for end nodes")

        sg = FibSwapGroup()
        sg.rank = entry.own_swap_rank
        sg.nodes = [entry.route[entry.own_idx]]

        sg.l_neigh, l_rank, l_most = sg._extend_1d(entry, range(entry.own_idx - 1, -1, -1))
        sg.own_idx = len(sg.nodes) - 1
        sg.nodes.reverse()
        sg.r_neigh, r_rank, r_most = sg._extend_1d(entry, range(entry.own_idx + 1, route_len))

        sg.l_adj = sg.l_neigh if l_most else sg.nodes[sg.own_idx - 1]
        sg.r_adj = sg.r_neigh if r_most else sg.nodes[sg.own_idx + 1]

        if l_rank == r_rank or entry.purif.get(f"{sg.l_neigh}-{sg.r_neigh}", 0) > 0:
            sg.dir = "b"
        elif l_rank < r_rank:
            sg.dir = "l"
        else:
            sg.dir = "r"

        return sg

    def _extend_1d(self, entry: FibEntry, index_range: Iterable[int]) -> tuple[str, int, bool]:
        own_is_boundary = True
        for i in index_range:
            node_rank = entry.swap[i]
            if node_rank > self.rank:
                return entry.route[i], entry.swap[i], own_is_boundary
            if node_rank == self.rank:
                own_is_boundary = False
                self.nodes.append(entry.route[i])
        raise ValueError(f"FibSwapGroup cannot find boundary in {entry.swap} from node index {entry.own_idx}")

    @property
    def l_most(self) -> bool:
        """Determine if own node is the leftmost node in the group."""
        return self.own_idx == 0

    @property
    def r_most(self) -> bool:
        """Determine if own node is the rightmost node in the group."""
        return self.own_idx == len(self.nodes) - 1

    def __repr__(self) -> str:
        tokens = [
            "FibSwapGroup(",
            self.l_neigh,
            "<" if self.dir in ("l", "b") else "~",
            "-".join(f"[{n}]" if i == self.own_idx else n for (i, n) in enumerate(self.nodes)),
            ">" if self.dir in ("b", "r") else "~",
            self.r_neigh,
            f", rank={self.rank})",
        ]
        return "".join(tokens)


class FibRequestGroup:
    """FIB information grouped by req_id."""

    def __init__(self, entry: FibEntry):
        """Construct from first FIB entry."""
        self.req_id = entry.req_id
        self.src = entry.route[0]
        self.dst = entry.route[-1]
        self.path_ids = {entry.path_id}

    def add(self, entry: FibEntry) -> None:
        """Check consistency and save FIB entry."""
        assert self.req_id == entry.req_id
        assert self.src == entry.route[0]
        assert self.dst == entry.route[-1]
        self.path_ids.add(entry.path_id)

    def remove(self, entry: FibEntry) -> bool:
        """
        Remove FIB entry.

        Returns:
            Whether the group has become empty.
        """
        assert self.req_id == entry.req_id
        self.path_ids.remove(entry.path_id)
        return len(self.path_ids) == 0


class Fib:
    def __init__(self):
        self.table: dict[int, FibEntry] = {}
        """
        FIB table.
        Key is path_id.
        Value is FIB entry.
        """
        self.by_req_id: dict[int, FibRequestGroup] = {}
        """
        Lookup table indexed by req_id.
        Key is req_id.
        Value contains aggregated information.
        """

    def get(self, path_id: int) -> FibEntry:
        """
        Retrieve an entry by path_id.

        Raises:
            IndexError: Entry not found.
        """
        try:
            return self.table[path_id]
        except KeyError:
            raise IndexError(f"FIB entry not found for path_id={path_id}")

    def insert_or_replace(self, entry: FibEntry):
        """
        Insert an entry or replace entry with same path_id.
        """
        self.erase(entry.path_id)
        self.table[entry.path_id] = entry

        rg = self.by_req_id.get(entry.req_id)
        if rg:
            rg.add(entry)
        else:
            rg = FibRequestGroup(entry)
            self.by_req_id[rg.req_id] = rg

    def erase(self, path_id: int):
        """
        Remove an entry from the table.

        Nonexistent entry is silent ignored.
        """
        try:
            entry = self.table.pop(path_id)
        except KeyError:
            return

        rg = self.by_req_id[entry.req_id]
        if rg.remove(entry):
            del self.by_req_id[entry.req_id]

    def list_path_ids_by_request_id(self, request_id: int) -> Set[int]:
        rg = self.by_req_id.get(request_id)
        if rg:
            return rg.path_ids
        return set()

    def find_request(self, predicate: Callable[[FibRequestGroup], bool]) -> Iterator[FibRequestGroup]:
        for rg in self.by_req_id.values():
            if predicate(rg):
                yield rg

    def __repr__(self):
        """Return a string representation of the forwarding table."""
        return "\n".join(
            f"Path ID: {path_id}, Request ID: {entry.req_id}, Path: {entry.route}, "
            f"Swap Sequence: {entry.swap}, Purification: {entry.purif}"
            for path_id, entry in self.table.items()
        )
