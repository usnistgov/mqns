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

from collections.abc import Callable, Iterator, Set
from dataclasses import dataclass
from typing import final

from mqns.network.proactive.message import SwapSequence
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

        To disable swapping, set swap_sequence to a list of zeros.

        When swapping is disabled, the forwarder will consume entanglement upon completing purification,
        without attempting entanglement swapping.

        Args:
            fib_entry: a FIB entry.
        """
        return self.swap[0] == 0 == self.swap[-1]

    def find_index_and_swap_rank(self, node_name: str) -> tuple[int, int]:
        """
        Determine the swapping rank of a node.

        Args:
            fib_entry: a FIB entry.
            node_name: a node name that exists in route.

        Returns:
            [0]: The node index in the route.
            [1]: Swapping rank of the node, explained in `PathInstructions`.

        Raises:
            IndexError - node does not exist in route.
        """
        idx = self.route.index(node_name)
        return idx, self.swap[idx]


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
            IndexError - Entry not found.
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
