"""
Test suite for simple data structure objects in forwarding.
"""

import pytest

from mqns.network.fw import parse_swap_sequence
from mqns.network.fw.fib import FibEntry, FibSwapGroup
from mqns.network.fw.message import validate_path_instructions


def test_parse_swap_sequence():
    """Test ``parse_swap_sequence`` function."""

    route3 = ["S", "R", "D"]
    route4 = ["S", "R1", "R2", "D"]
    route5 = ["S", "R1", "R2", "R3", "D"]
    route6 = ["S", "R1", "R2", "R3", "R4", "D"]
    route7 = ["S", "R1", "R2", "R3", "R4", "R5", "D"]

    assert parse_swap_sequence("disabled", route3) == [0, 0, 0]
    assert parse_swap_sequence("asap", route3) == [1, 0, 1]
    assert parse_swap_sequence("asap", route4) == [1, 0, 0, 1]
    assert parse_swap_sequence("l2r", route4) == [2, 0, 1, 2]
    assert parse_swap_sequence("r2l", route4) == [2, 1, 0, 2]
    assert parse_swap_sequence("asap", route5) == [1, 0, 0, 0, 1]
    assert parse_swap_sequence("baln", route5) == [2, 0, 1, 0, 2]
    assert parse_swap_sequence("l2r", route5) == [3, 0, 1, 2, 3]
    assert parse_swap_sequence("r2l", route5) == [3, 2, 1, 0, 3]
    assert parse_swap_sequence("asap", route6) == [1, 0, 0, 0, 0, 1]
    assert parse_swap_sequence("baln", route6) == [3, 0, 1, 0, 2, 3]
    assert parse_swap_sequence("baln2", route6) == [3, 2, 0, 1, 0, 3]
    assert parse_swap_sequence("l2r", route6) == [4, 0, 1, 2, 3, 4]
    assert parse_swap_sequence("r2l", route6) == [4, 3, 2, 1, 0, 4]
    assert parse_swap_sequence("asap", route7) == [1, 0, 0, 0, 0, 0, 1]
    assert parse_swap_sequence("baln", route7) == [3, 0, 1, 0, 2, 0, 3]
    assert parse_swap_sequence("baln2", route7) == [3, 0, 2, 0, 1, 0, 3]
    assert parse_swap_sequence("l2r", route7) == [5, 0, 1, 2, 3, 4, 5]
    assert parse_swap_sequence("r2l", route7) == [5, 4, 3, 2, 1, 0, 5]


def test_path_validation():
    """Test path validation logic."""

    route3 = ["A", "B", "C"]
    swap3 = [1, 0, 1]
    scut3 = [-1, 1000, -1]
    mv3 = [(1, 1)] * 2

    with pytest.raises(ValueError, match="route is empty"):
        validate_path_instructions({"req_id": 0, "route": [], "swap": [], "swap_cutoff": [], "purif": {}})

    with pytest.raises(ValueError, match="swapping order"):
        validate_path_instructions(
            {"req_id": 0, "route": ["A", "B", "C", "D", "E"], "swap": swap3, "swap_cutoff": scut3, "purif": {}}
        )

    with pytest.raises(ValueError, match="swap_cutoff"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": [-1, 1000, 1000, -1], "purif": {}}
        )

    with pytest.raises(ValueError, match="multiplexing vector"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": [(1, 1)] * 3, "purif": {}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"P-Q": 1}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"A-B-C": 1}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"B-B": 1}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"C-A": 1}}
        )


@pytest.mark.parametrize(
    ("purif", "own", "expected"),
    [
        # without purification
        (None, "A", None),
        (None, "B", ("A", "BC", "D", "r")),
        (None, "C", ("A", "BC", "D", "r")),
        (None, "D", ("A", "DF", "G", "r")),
        (None, "E", ("D", "E", "F", "b")),
        (None, "F", ("A", "DF", "G", "r")),
        (None, "G", ("A", "G", "J", "b")),
        (None, "H", ("G", "HI", "J", "l")),
        (None, "I", ("G", "HI", "J", "l")),
        (None, "J", None),
        # with valid purification
        ("A-G", "B", ("A", "BC", "D", "r")),
        ("A-G", "D", ("A", "DF", "G", "b")),
        ("A-G", "F", ("A", "DF", "G", "b")),
        ("A-G", "H", ("G", "HI", "J", "l")),
    ],
)
def test_fib_swap_group(purif: str | None, own: str, expected: tuple[str, str, str, str] | None):
    nodes = "ABCDEFGHIJ"
    ranks = "3001012003"
    entry = FibEntry(
        path_id=0,
        req_id=0,
        route=list(nodes),
        own_idx=nodes.index(own),
        swap=[int(v) for v in ranks],
        swap_cutoff=[None] * 9,
        purif={purif: 1} if purif else {},
    )

    if expected is None:
        with pytest.raises(ValueError, match="undefined for end nodes"):
            FibSwapGroup.compute(entry)
        return

    sg = entry.sg
    assert sg.nodes == list(expected[1])
    assert sg.own_idx == sg.nodes.index(own)
    assert (sg.l_neigh, "".join(sg.nodes), sg.r_neigh, sg.dir) == expected

    _, rank = entry.find_index_and_swap_rank(sg.nodes[0])
    assert sg.rank == rank
