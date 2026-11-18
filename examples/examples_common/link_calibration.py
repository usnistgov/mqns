"""
Interface with ``examples/linear_attempts.py`` for link calibration.
"""

from collections.abc import Sequence

from mqns.entity.qchannel import LinkArchLiteral


def make_linear_attempts_command(
    *,
    runs=1,
    sim_duration=1.0,
    ch_lengths: Sequence[float] = [32, 18],
    link_arch: LinkArchLiteral,
    output_csv: str,
) -> str:
    L = " ".join(str(length) for length in ch_lengths)
    return (
        f"python linear_attempts.py --runs {runs} --sim_duration {sim_duration}"
        f" --L {L} --M 1 --link_arch {link_arch}"
        f" --csv {output_csv}"
    )
