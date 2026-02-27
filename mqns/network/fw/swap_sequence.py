from typing import Literal

from mqns.network.fw.message import SwapSequence

type SwapPolicy = Literal["disabled", "asap", "l2r", "r2l", "baln", "baln2"]
"""
Swap policies.

* "disabled": swapping disabled.
* "asap": as soon as possible.
* "l2r": left to right.
* "r2l": right to left.
* "baln": balanced tree.
* "baln2": balanced tree, mirrored.
"""

type SwapSequenceInput = SwapSequence | SwapPolicy


def _l2r(n: int) -> SwapSequence:
    return [n] + list(range(n + 1))


def _baln(n: int) -> SwapSequence:
    if n == 0:
        return [0, 0]
    so = [0] * (n + 1)
    nodes = list(range(1, n + 1))
    rank = 0
    while nodes:
        level = nodes[::2]
        for node_id in level:
            so[node_id] = rank
        nodes = nodes[1::2]
        rank += 1
    terminal_rank = max(so) + 1
    return [terminal_rank] + so[1:] + [terminal_rank]


def parse_swap_sequence(input: SwapSequenceInput, route: list[str]) -> SwapSequence:
    """
    Parse swap sequence input.

    Args:
        input: Either an explicitly specified swap sequence or a swap policy.
        route: List of nodes in the path.

    Returns:
        The swap sequence.

    Raises:
        IndexError - a predefined swap sequence is requested but not defined.
        ValueError - specified or retrieved swap sequence does not match the route length.
    """
    if isinstance(input, list):
        swap = input
        if len(swap) != len(route):
            raise ValueError(f"swap sequence {swap} does not match route {route} with {len(route)} nodes")
        return swap

    n = len(route) - 2
    match input:
        case "disabled":
            swap = [0] * (n + 2)
        case "asap":
            swap = [1] + [0] * n + [1]
        case "l2r":
            swap = _l2r(n)
        case "r2l":
            swap = _l2r(n)
            swap.reverse()
        case "baln":
            swap = _baln(n)
        case "baln2":
            swap = _baln(n)
            swap.reverse()
        case _:
            raise IndexError(f"unknown swap policy {input}")
    return swap
