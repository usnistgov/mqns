import math
from typing import cast

from mqns.entity.base_channel import default_light_speed
from mqns.network.proactive.message import SwapSequence

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
    # for 4-repeater
    "swap_4_asap": [1, 0, 0, 0, 0, 1],
    "swap_4_baln": [3, 0, 1, 0, 2, 3],
    "swap_4_baln2": [3, 2, 0, 1, 0, 3],
    "swap_4_l2r": [4, 0, 1, 2, 3, 4],
    "swap_4_r2l": [4, 3, 2, 1, 0, 4],
    # for 5-repeater example
    "swap_5_asap": [1, 0, 0, 0, 0, 0, 1],
    "swap_5_baln": [3, 0, 1, 0, 2, 0, 3],  # need to specify exact doubling  => this is used in the vora paper
    "swap_5_baln2": [3, 0, 2, 0, 1, 0, 3],
    "swap_5_l2r": [5, 0, 1, 2, 3, 4, 5],
    "swap_5_r2l": [5, 4, 3, 2, 1, 0, 5],
}


def parse_swap_sequence(input: SwapSequence | str, route: list[str]) -> SwapSequence:
    """
    Parse swap sequence input.

    Args:
        input: Either an explicitly specified swap sequence,
               or a string that identifies either a predefined swap sequence or a swapping policy.
               The swapping policy may be one of: asap, baln, l2r, r2l.
        route: List of nodes in the path.

    Returns:
        The swap sequence.

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


def compute_vora_swap_sequence(
    *,
    lengths: list[float],
    attempts: list[float],
    success: list[float],
    ps=0.5,
    t_cohere=0.010,
    qchannel_capacity=25,
) -> SwapSequence:
    """
    Compute vora swapping order from training data.
    The training data should be generated with qchannel_capacity=1 and swapping disabled.
    See https://doi.org/10.48550/arXiv.2504.14040 for description of vora algorithm.

    Args:
        lengths: qchannel lengths on a linear path, in kilometers.
        attempts: mean value of attempts per second on each qchannel (from training data).
        success: mean value of success ratio on each qchannel (from training data).
        ps: probability of successful swapping.
        t_cohere: memory coherence time, in seconds.
        qchannel_capacity: qchannel capacity.
    """
    from mqns.network.proactive.vora_utils import get_Bq, merge_close, voraswap  # noqa: PLC0415

    assert len(lengths) == len(attempts) == len(success) >= 1
    assert min(lengths) > 0
    assert min(attempts) > 0
    assert 1 >= max(success) >= min(success) >= 0

    # Gather characteristics of each quantum channel.
    L_list = merge_close(lengths, 0.50)
    C0 = merge_close(attempts, 0.50)
    P = merge_close(success, 0.75)

    # Compute time slot for the external phase i.e. elementary entanglement establishment.
    T_cutoff = t_cohere
    tau = 2 * sum(L_list) / default_light_speed  # for heralding
    T_ext = T_cutoff - tau  # for external phase

    # Derive actual capacity (#attempts/time_slot) passed to VoraSwap algorithm.
    C = [round(c * qchannel_capacity * T_ext) for c in C0]

    # Compute Binomial coefficients, large enough for the computation.
    # This function is slow but it has internal file-based caching.
    Bq = get_Bq(math.ceil(max(C) / 1e3) * 1000, ps)

    # Invoke VoraSwap algorithm.
    result = voraswap(C, P, ps, Bq=Bq, Ts=T_cutoff)
    assert type(result) is dict

    # Convert to MQNS swap sequence.
    # result["order"] lists the 1-based identifier of repeater in the order of swapping.
    # For example, [1,3,2] means: R1 swaps first, R3 swaps next, R2 swaps last.
    # SwapSequence lists the swapping rank of each node sorted by linear path,
    # and also includes high numbers for source and destination nodes,
    # so that the same swapping order should be written as:
    #    [3,  0,   2,   1,   3]
    #     S   R1   R2   R3   D
    so = [0] * (1 + len(lengths))
    for i, j in enumerate(cast(list[int], result["order"])):
        so[j] = i
    so[0] = so[-1] = 1 + max(so)
    return so
