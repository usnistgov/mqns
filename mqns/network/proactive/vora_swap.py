import math
from typing import cast

from mqns.entity.base_channel import default_light_speed
from mqns.network.fw.message import SwapSequence


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
    tau = 2 * sum(L_list) / default_light_speed[0]  # for heralding
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
