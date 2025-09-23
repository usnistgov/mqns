_predefined_swap_sequences = {
    # disable swapping (for studying isolated links)
    "no_swap": [0, 0, 0],
    # for 1-repeater
    "swap_1": [1, 0, 1],
    "swap_1_asap": [1, 0, 1],
    # for 2-repeater
    "swap_2_asap": [1, 0, 0, 1],
    "swap_2_l2r": [2, 0, 1, 2],
    "swap_2_r2l": [2, 1, 0, 2],
    # for 3-repeater
    "swap_3_asap": [1, 0, 0, 0, 1],
    "swap_3_baln": [2, 0, 1, 0, 2],
    "swap_3_l2r": [3, 0, 1, 2, 3],
    "swap_3_r2l": [3, 2, 1, 0, 3],
    "swap_3_vora_uniform": [3, 0, 2, 1, 3],  # equiv. [2,0,1,0,2] ~ baln
    "swap_3_vora_increasing": [3, 0, 1, 2, 3],
    "swap_3_vora_decreasing": [3, 2, 1, 0, 3],
    "swap_3_vora_mid_bottleneck": [3, 1, 2, 0, 3],  # [2,0,1,0,2]  ~ baln
    # for 4-repeater
    "swap_4_asap": [1, 0, 0, 0, 0, 1],
    "swap_4_baln": [3, 0, 1, 0, 2, 3],
    "swap_4_baln2": [3, 2, 0, 1, 0, 3],
    "swap_4_l2r": [4, 0, 1, 2, 3, 4],
    "swap_4_r2l": [4, 3, 2, 1, 0, 4],
    "swap_4_vora_uniform": [4, 0, 3, 1, 2, 4],  # equiv. [3,0,2,0,1,3]
    "swap_4_vora_increasing": [4, 0, 1, 3, 2, 4],  # equiv. [3,0,1,2,0,3]
    "swap_4_vora_decreasing": [4, 3, 1, 2, 0, 4],  # equiv. [3,2,0,1,0,3]
    "swap_4_vora_mid_bottleneck": [4, 0, 2, 3, 1, 4],  # equiv. [3,0,1,2,0,3]
    "swap_4_vora_uniform2": [3, 0, 2, 0, 1, 3],
    "swap_4_vora_increasing2": [3, 0, 1, 2, 0, 3],
    "swap_4_vora_decreasing2": [3, 2, 0, 1, 0, 3],
    "swap_4_vora_mid_bottleneck2": [3, 0, 1, 2, 0, 3],
    # for 5-repeater example
    "swap_5_asap": [1, 0, 0, 0, 0, 0, 1],
    "swap_5_baln": [3, 0, 1, 0, 2, 0, 3],  # need to specify exact doubling  => this is used in the vora paper
    "swap_5_baln2": [3, 0, 2, 0, 1, 0, 3],
    "swap_5_l2r": [5, 0, 1, 2, 3, 4, 5],
    "swap_5_r2l": [5, 4, 3, 2, 1, 0, 5],
    "swap_5_vora_uniform": [5, 0, 3, 1, 4, 2, 5],  # [3,0,1,0,2,0,3]  ~ baln
    "swap_5_vora_increasing": [5, 0, 3, 1, 4, 2, 5],  # [3,0,1,0,2,0,3] ~ baln
    "swap_5_vora_decreasing": [5, 2, 4, 1, 3, 0, 5],  # [3,0,2,0,1,0,3] ~ baln2
    "swap_5_vora_mid_bottleneck": [5, 0, 4, 2, 3, 1, 5],  # [3,0,2,0,1,0,3] ~ baln2
}


def parse_swap_sequence(input: list[int] | str, route: list[str]) -> list[int]:
    """
    Parse swapping order input.

    Args:
        input: Either an explicitly specified swapping order,
               or a string that identifies either a predefined swapping order or a swapping policy.
               The swapping policy may be one of: asap, baln, l2r, r2l.
        route: List of nodes in the path.

    Returns:
        The swapping order.

    Raises:
        IndexError - a predefined swap sequence is requested but not defined.
        ValueError - specified or retrieved swap sequence does not match the route length.
    """
    if isinstance(input, list):
        swap = input
    else:
        try:
            swap = _predefined_swap_sequences[input]
        except KeyError:
            try:
                swap = _predefined_swap_sequences[f"swap_{len(route) - 2}_{input}"]
            except KeyError:
                raise IndexError(f"swap sequence {input} undefined for {len(route)} nodes")

    if len(swap) != len(route):
        raise ValueError(f"swap sequence {swap} does not match route {route} with {len(route)} nodes")
    return swap
