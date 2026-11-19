"""
``CutoffSchemeWaitTime`` evaluation in a linear topology.

1. Generate a sample parameter definition file::

    # sample parameters in TOML
    python linear_wait.py --p +toml >lw.p.toml
    # sample parameters in JSON
    python linear_wait.py --p +json >lw.p.json
    # JSON schema of parameters
    python linear_wait.py --p +schema >lw.p.schema.json

2. Modify the parameter definition file and fill in these fields:
   sim_duration, link_arch, ch_lengths.

3. Generate a command line for link calibration::

    python linear_wait.py --p lw.p.toml --runs 1000 --calibration

4. Run link calibration using the generated command line.

5. Modify the parameter definition file and fill in all other fields.
   Modeling software for this step is not yet released.

6. Run path evaluation::

    python linear_wait.py --p lw.p.toml --runs 100 --json lw.r.json
"""

import itertools
import json
import sys
from multiprocessing import Pool, freeze_support
from pathlib import Path
from typing import Annotated, Self, TypedDict, override

import numpy as np
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, model_validator
from tap import Tap

from mqns.entity.qchannel import LinkArchLiteral
from mqns.network.builder import CTRL_DELAY, ChannelParam, NetworkBuilder, NodeDef, tap_configure
from mqns.network.protocol.consumer import RequestCounters
from mqns.simulator import Simulator
from mqns.utils import json_default, log, rng, seed_seq_env, unwrap

from examples_common.link_calibration import make_linear_attempts_command
from examples_common.param_def import load_param_def

log.set_default_level("CRITICAL")


class Args(Tap):
    p: str = ""  # scenario definition file
    calibration: bool = False  # write link calibration script to stdout
    workers: int = 1  # number of workers for parallel execution
    runs: int = 10  # number of trials per parameter set
    json: str = ""  # save details as JSON file

    @override
    def configure(self) -> None:
        tap_configure(self)


class ParamDef(BaseModel):
    sim_duration: PositiveFloat
    """Simulation duration in seconds."""
    link_arch: LinkArchLiteral
    """Link architecture, either DIM-dual or DIM-BK."""
    ch_lengths: list[PositiveFloat]
    """
    Channel lengths in kilometers.
    This determines number of links, which then derives number of nodes and number of repeaters.
    """
    ch_fidelity: list[tuple[PositiveFloat, NonNegativeFloat, NonNegativeFloat, NonNegativeFloat]]
    """Initial fidelity including channel errors but excluding memory errors."""
    t_cohere: list[PositiveFloat]
    """Memory coherence time per node, which is inverse of memory dephasing rate."""
    mem_alloc: list[PositiveInt]
    """Memory allocation per channel per node."""
    swap_rank: list[NonNegativeInt]
    """
    Swap ranks per node, see :py:class:`mqns.network.fw.message.SwapSequence`.
    Set to ``[1, 0, .., 0, 1]`` for ASAP.
    Otherwise, repeater with smaller rank swaps before adjacent repeater with greater rank,
    and end nodes must have the highest ranks.
    """
    swap_wait: list[PositiveFloat]
    """
    Wait-time cutoff per repeater per channel.
    Set to a value greater than t_cohere to effectively disable wait-time cutoff.
    """
    p_swap: Annotated[float, Field(ge=0.5, le=1.0)]
    """Success probability of memory-memory swap."""
    swap_delay: NonNegativeFloat
    """
    Duration of memory-memory swap in seconds.
    Memory dephasing continues during the swap.
    """

    @property
    def n_links(self) -> int:
        return len(self.ch_lengths)

    @property
    def n_nodes(self) -> int:
        return self.n_links + 1

    @property
    def n_repeaters(self) -> int:
        return self.n_links - 1

    @model_validator(mode="after")
    def _(self) -> Self:
        def check_length(key: str, n: int) -> None:
            if len(getattr(self, key)) != n:
                raise ValueError(f"--{key} should have {n} elements")

        check_length("ch_fidelity", self.n_links)
        check_length("t_cohere", self.n_nodes)
        check_length("mem_alloc", 2 * self.n_links)
        check_length("swap_rank", self.n_nodes)
        check_length("swap_wait", 2 * self.n_repeaters)

        return self


def make_param_sample() -> ParamDef:
    # Suppose the topology is A-B-C-D, see below comments for how everything is mapped.
    return ParamDef(
        sim_duration=1,
        link_arch="DIM-dual",
        ch_lengths=[32, 18, 10],  # AB, BC, CD links
        ch_fidelity=[(1.0, 0.0, 0.0, 0.0)] * 3,  # AB, BC, CD links
        t_cohere=[0.01, 0.01, 0.01, 0.01],  # A, B, C, D nodes
        mem_alloc=[1, 1, 1, 1, 1, 1],  # A-right, B-left, B-right, C-left, C-right, D-left
        swap_rank=[2, 0, 1, 2],  # B swaps before C, A and D have highest rank because they are end nodes
        swap_wait=[1, 1, 1, 1],  # B-left, B-right, C-left, C-right
        p_swap=0.5,
        swap_delay=0,
    )


class Result(TypedDict):
    seed: int
    """Random seed used in this run."""
    count: int
    """Total quantity of EPRs delivered to both end nodes."""
    rate: float
    """End-to-end rate."""
    fid_mean: float
    """Fidelity mean."""
    fid_std: float
    """Fidelity standard deviation."""
    fid_values: list[float]
    """Fidelity values as they are consumed by the second end node."""


def run_simulation(p: ParamDef, seed: int) -> Result:
    rng.reseed(seed)

    net = (
        NetworkBuilder(
            epr_type="M",
        )
        .topo_linear(
            nodes=[
                NodeDef(f"node{i}", mem_capacity=sum(node_mem), t_cohere=t_cohere)
                for i, t_cohere, node_mem in zip(
                    range(p.n_nodes), p.t_cohere, itertools.batched([0, *p.mem_alloc, 0], n=2, strict=True), strict=True
                )
            ],
            channels=[ChannelParam(ch_length=l, init_fidelity=fid) for l, fid in zip(p.ch_lengths, p.ch_fidelity, strict=True)],
            link_arch=p.link_arch,
            fiber_error="PERFECT",  # fiber_error is precomputed in init_fidelity
            bsa_error="PERFECT",  # bsa_error is precomputed in init_fidelity
        )
        .proactive_centralized(
            p_swap=p.p_swap,
            swap_delay=p.swap_delay,
            swap_error="PERFECT",  # no error applied by the swap gates
            swap_error_at="f",  # memory decoherence continues during swap
        )
        .request(
            src_dst := ("node0", f"node{p.n_nodes - 1}"),
            req_id=0,
            bufferspace_mv=p.mem_alloc,
            swap=p.swap_rank,
            swap_cutoff=p.swap_wait,
        )
        .make_network()
    )

    RequestCounters.enable_collect_all(net, 0, src_dst)
    s = Simulator(0, p.sim_duration + CTRL_DELAY, install_to=(log, net))
    s.run()

    req_cnt = RequestCounters.of(net, 0, src_dst)
    fid_values = unwrap(req_cnt.consumed_fidelity_values)
    return Result(
        seed=seed,
        count=req_cnt.n_consumed,
        rate=req_cnt.get_rate(p.sim_duration),
        fid_mean=req_cnt.consumed_avg_fidelity,
        fid_std=np.std(unwrap(req_cnt.consumed_fidelity_values)).item(),
        fid_values=fid_values,
    )


def main(args: Args) -> None:
    p = load_param_def(args.p, make_param_sample())
    if p is None:
        return

    if args.calibration:
        print(
            make_linear_attempts_command(
                runs=args.runs,
                ch_lengths=p.ch_lengths,
                link_arch=p.link_arch,
                output_csv=str(Path(args.p).with_suffix(".lam.csv")),
            )
        )
        return

    with Pool(processes=args.workers) as pool:
        results = pool.starmap(run_simulation, itertools.product([p], seed_seq_env(args.runs, 0)))

    if args.json:
        with open(args.json, "w") as file:
            json.dump([p, *results], file, default=json_default)


if __name__ == "__main__":
    freeze_support()
    if sys.version_info[:3] <= (3, 13.0):
        # Python 3.13 is needed for itertools.batched(strict=True) option.
        raise SystemError("This scenario requires Python 3.13 or later.")
    args = Args().parse_args()
    main(args)
