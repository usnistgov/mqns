from typing import Literal, TypedDict


class LinkStateEntry(TypedDict):
    node: str
    """Node name of the node sending link state report."""
    neighbor: str
    """Node name of the other node sharing an entanglement with this node."""
    qubit: str
    """Reservation key of the qubit."""


class LinkStateMsg(TypedDict):
    cmd: Literal["LS"]
    ls: list[LinkStateEntry]
