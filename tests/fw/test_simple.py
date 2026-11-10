"""
Test suite for simple data structure objects in forwarding.
"""

import pytest

from mqns.network.fw import FibPath, PathInstructions, parse_swap_sequence, validate_path_instructions


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

    inst_base: PathInstructions = {"path_id": 0, "route": ["A", "B", "C"], "swap": [1, 0, 1], "purif": {}}

    validate_path_instructions(inst_base)

    # path_id
    with pytest.raises(ValueError, match="path_id"):
        validate_path_instructions({**inst_base, "path_id": -1})

    # route
    with pytest.raises(ValueError, match="route is empty"):
        validate_path_instructions({**inst_base, "route": []})

    # ll_dir
    validate_path_instructions({**inst_base, "ll_dir": "LR"})
    with pytest.raises(ValueError, match="ll_dir is invalid or does not match route length"):
        validate_path_instructions({**inst_base, "ll_dir": "RRR"})
    with pytest.raises(ValueError, match="ll_dir is invalid or does not match route length"):
        validate_path_instructions({**inst_base, "ll_dir": "DIR"})

    # bufferspace_mv
    validate_path_instructions({**inst_base, "bufferspace_mv": [1, 1] * 2}, bufferspace=True)

    with pytest.raises(ValueError, match="bufferspace_mv must be absent"):
        validate_path_instructions({**inst_base, "bufferspace_mv": [1, 1] * 2})

    with pytest.raises(ValueError, match="bufferspace_mv must be present"):
        validate_path_instructions({**inst_base}, bufferspace=True)

    with pytest.raises(ValueError, match="bufferspace_mv does not match route length"):
        validate_path_instructions({**inst_base, "bufferspace_mv": [1, 1] * 3}, bufferspace=True)

    with pytest.raises(ValueError, match="bufferspace_mv must be positive"):
        validate_path_instructions({**inst_base, "bufferspace_mv": [4, 2, 3, 0]}, bufferspace=True)

    # reactive_qubits
    validate_path_instructions({**inst_base, "reactive_qubits": ["epr0", "epr1"]}, reactive=True)

    with pytest.raises(ValueError, match="reactive_qubits must be absent"):
        validate_path_instructions({**inst_base, "reactive_qubits": ["epr0", "epr1"]})

    with pytest.raises(ValueError, match="reactive_qubits must be present"):
        validate_path_instructions({**inst_base}, reactive=True)

    with pytest.raises(ValueError, match="reactive_qubits does not match route length"):
        validate_path_instructions({**inst_base, "reactive_qubits": ["epr0", "epr1", "epr2"]}, reactive=True)

    # swap and swap_cutoff
    validate_path_instructions({**inst_base, "swap_cutoff": [2000, 1000]})

    with pytest.raises(ValueError, match="swapping order"):
        validate_path_instructions({**inst_base, "swap": [1, 0, 0, 1]})

    with pytest.raises(ValueError, match="swap_cutoff"):
        validate_path_instructions({**inst_base, "swap_cutoff": [2000, 1000, 1000]})

    # purif
    validate_path_instructions({**inst_base, "purif": {"A-B": 2, "B-C": 1, "A-C": 1}})

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions({**inst_base, "purif": {"P-Q": 1}})

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions({**inst_base, "purif": {"A-B-C": 1}})

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions({**inst_base, "purif": {"B-B": 1}})

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions({**inst_base, "purif": {"C-A": 1}})


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
    fp = FibPath(
        path_id=0,
        route=nodes,
        own_idx=nodes.index(own),
        swap=[int(v) for v in ranks],
        swap_cutoff=[None] * 9,
        purif={purif: 1} if purif else {},
    )

    sg = fp.sg

    if expected is None:
        assert fp.own_is_end_node
        assert sg is None
        return

    assert not fp.own_is_end_node
    assert sg is not None
    assert sg.nodes == list(expected[1])
    assert sg.own_idx == sg.nodes.index(own)
    assert (sg.l_neigh, "".join(sg.nodes), sg.r_neigh, sg.dir) == expected
    assert sg.rank == fp.swap[fp.route.index(sg.nodes[0])]
