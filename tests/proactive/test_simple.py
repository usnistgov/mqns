"""
Test suite for simple data structure objects in proactive forwarding.
"""

import pytest

from mqns.network.proactive.message import validate_path_instructions


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
