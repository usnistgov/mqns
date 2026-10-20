from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import overload

from mqns.utils import rng, unwrap


class TopoGen(ABC):
    """
    Topology generator for :py:class:`~mqns.network.builder.NetworkBuilder`.
    """

    @abstractmethod
    def build_topo(
        self,
        *,
        make_nodes: Callable[[int], Sequence[str]],
        make_link: Callable[[int | str, int | str], None],
    ) -> None: ...


class GridTopoGen(TopoGen):
    """
    Grid topology generator.
    """

    def __init__(self, shape: tuple[int, int]):
        """
        Args:
            shape: Number of rows and columns, both must be positive integer.
        """
        if min(shape) < 1:
            raise ValueError("every dimension of a grid topology must be positive")
        self.shape = shape

    def build_topo(self, *, make_nodes, make_link) -> None:
        rows, cols = self.shape
        make_nodes(rows * cols)

        # horizontal links
        for r in range(rows):
            idx = r * cols
            for c in range(cols - 1):
                make_link(idx + c, idx + c + 1)

        # vertical links
        for r in range(rows - 1):
            idx = r * cols
            for c in range(cols):
                make_link(idx + c, idx + c + cols)


class LinearTopoGen(GridTopoGen):
    """
    Linear topology generator.
    """

    def __init__(self, n: int):
        """
        Args:
            n: Number of nodes.
        """
        super().__init__((1, n))


class TreeTopoGen(TopoGen):
    """
    Tree topology generator.
    """

    @overload
    def __init__(self, *, size: int, arity=2):
        """
        Args:
            size: Number of nodes.
            arity: Maximum number of children per node.
        """

    @overload
    def __init__(self, *, levels: int, arity=2):
        """
        Args:
            levels: Number of node levels from the root to each leaf.
            arity: Number of children per node.
        """

    def __init__(self, *, size: int | None = None, levels: int | None = None, arity=2):
        if arity < 0:
            raise ValueError("arity must be nonnegative")
        self.arity = arity

        if arity == 0:
            if size in (1, None) and levels in (1, None):
                self.size = 1
                return
            else:
                raise ValueError("a tree with arity 0 can only have size=1 or levels=1")

        if size is None:
            levels = unwrap(levels)
            if levels < 1:
                raise ValueError("levels must be positive")
            self.size = sum(arity**i for i in range(levels))
        elif size < 1:
            raise ValueError("size must be positive")
        else:
            self.size = size

    def build_topo(self, *, make_nodes, make_link) -> None:
        make_nodes(self.size)
        if self.size == 1:  # including but not limited to arity=0
            return

        # Node i can only be a parent if its first child index is less than tree size.
        # Solve for i in `i * self.arity + 1 < self.size` yields:
        max_parent_idx = (self.size - 2) // self.arity
        for i in range(max_parent_idx + 1):
            first_child = i * self.arity + 1
            for j in range(first_child, min(first_child + self.arity, self.size)):
                make_link(i, j)


class RandomTopoGen(TopoGen):
    """
    Random topology generator.
    """

    def __init__(self, size: int, edges: int):
        """
        Args:
            size: Number of nodes.
            edges: Number of quantum channels.
        """
        if size < 1:
            raise ValueError("size must be positive")
        if edges < size - 1:
            raise ValueError("too few edges to ensure connectivity")
        if edges > size * (size - 1) // 2:
            raise ValueError("too many edges")
        self.size = size
        self.edges = edges

    def build_topo(self, *, make_nodes, make_link) -> None:
        make_nodes(self.size)

        edges = set[tuple[int, int]]()

        # Ensure connectivity of every node.
        for b in range(1, self.size):
            a = rng.integers(0, b, dtype=int)
            make_link(a, b)
            edges.add((a, b))

        # Create additional links.
        while len(edges) < self.edges:
            while True:
                a = rng.integers(0, self.size, dtype=int)
                b = rng.integers(0, self.size, dtype=int)
                key = (a, b) if a < b else (b, a)
                if a != b and key not in edges:
                    break
            # The b,a order is intentional, for matching the behavior of RandomTopology as of commit 0f59354f.
            make_link(b, a)
            edges.add(key)

        assert len(edges) == self.edges
