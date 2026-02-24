from typing import Literal, TypedDict


class LinkStateEntry(TypedDict):
    node: str
    neighbor: str
    qubit: int


class LinkStateMsg(TypedDict):
    cmd: Literal["LS"]
    ls: list[LinkStateEntry]
