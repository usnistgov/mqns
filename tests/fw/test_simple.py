"""
Test suite for simple data structure objects in forwarding.
"""

import pytest

from mqns.network.fw import parse_swap_sequence
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

    route3 = ["n1", "n2", "n3"]
    swap3 = [1, 0, 1]
    scut3 = [-1, 1000, -1]
    mv3 = [(1, 1)] * 2

    with pytest.raises(ValueError, match="route is empty"):
        validate_path_instructions({"req_id": 0, "route": [], "swap": [], "swap_cutoff": [], "purif": {}})

    with pytest.raises(ValueError, match="swapping order"):
        validate_path_instructions(
            {"req_id": 0, "route": ["n1", "n2", "n3", "n4", "n5"], "swap": swap3, "swap_cutoff": scut3, "purif": {}}
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
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"r1-r2": 1}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"n1-n2-n3": 1}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"n2-n2": 1}}
        )

    with pytest.raises(ValueError, match="purif segment"):
        validate_path_instructions(
            {"req_id": 0, "route": route3, "swap": swap3, "swap_cutoff": scut3, "m_v": mv3, "purif": {"n3-n1": 1}}
        )
