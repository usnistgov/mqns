"""
This script simulates a linear topology with ``CutoffSchemeWaitTime``.
The controller sets wait-time budgets so that the end-to-end EPRs are delivered with a certain minimal fidelity.
"""

import csv
import itertools
import json
from dataclasses import dataclass
from multiprocessing import Pool, freeze_support
from typing import TypedDict, override

import numpy as np
from tap import Tap

from mqns.entity.qchannel import LinkArchDimDual
from mqns.models.epr import MixedStateEntanglement
from mqns.network.builder import CTRL_DELAY, ChannelParam, NetworkBuilder, NodeDef, tap_configure
from mqns.network.fw import ForwarderConsumeCounters
from mqns.simulator import Simulator
from mqns.utils import log, rng

log.set_default_level("CRITICAL")


class Args(Tap):
    workers: int = 1  # number of workers for parallel execution
    runs: int = 10  # number of trials per parameter set
    sim_duration: float = 1.0  # simulation duration in seconds
    json: str = ""  # save stats as JSON file
    csv: str = ""  # save stats summary as CSV file

    @override
    def configure(self) -> None:
        tap_configure(self)


SIMULATOR_ACCURACY = 1000000
SEED_BASE = 100


@dataclass
class RowInput:
    f_req: float
    """Required fidelity."""
    w: tuple[float, float]
    """Wait-time budgets."""


ROW_INPUTS: list[RowInput] = [
    RowInput(0.75, (0.031731, 0.023798)),
    RowInput(0.7823, (0.023991, 0.017993)),
    RowInput(0.8146, (0.017056, 0.012792)),
    RowInput(0.8469, (0.010775, 0.008081)),
    RowInput(0.8791, (0.005053, 0.003789)),
    RowInput(0.8904, (0.003155, 0.002367)),
    RowInput(0.9001, (0.001569, 0.001177)),
    RowInput(0.9098, (1.9e-05, 1.4e-05)),
]


class Stats(TypedDict):
    count: int
    """Rate -- number of entanglements consumed."""
    fid: list[float]
    """Fidelity -- individual fidelity values."""


class Row(TypedDict):
    f_req: float
    """Required fidelity."""
    details: list[Stats]
    """Per-run statistics."""


def convert_fidelity(raw_fidelity: float, x_ratio: float, y_ratio: float, z_ratio: float):
    p_error = 1 - raw_fidelity
    return raw_fidelity, p_error * z_ratio, p_error * x_ratio, p_error * y_ratio


def run_simulation(seed: int, args: Args, ri: RowInput) -> Stats:
    rng.reseed(seed)

    net = (
        NetworkBuilder(
            epr_type=MixedStateEntanglement,
        )
        .topo_linear(
            nodes=[
                NodeDef("S", t_cohere=1 / 5),
                "R",
                "D",
            ],
            t_cohere=1 / 10,
            channels=[
                ChannelParam(ch_length=32, init_fidelity=convert_fidelity(0.9474, 0.1427, 0.1427, 0.7147)),
                ChannelParam(ch_length=18, init_fidelity=convert_fidelity(0.9677, 0.1547, 0.1547, 0.6907)),
            ],
            fiber_alpha=0.2,
            link_arch=LinkArchDimDual,
            eta_d=0.58,
            eta_s=0.99,
            frequency=80e6,
            tau_0=10e-6,
        )
        .proactive_centralized(
            p_swap=0.5,
            swap_delay=340e-6,
            swap_error=None,  # TODO memory decoherence should continue during swap
        )
        .request(
            "S-D",
            swap_cutoff=ri.w,
        )
        .make_network()
    )

    ForwarderConsumeCounters.enable_collect_all_on_path(net, "S", "D")

    s = Simulator(0, args.sim_duration + CTRL_DELAY, accuracy=SIMULATOR_ACCURACY, install_to=(log, net))
    s.run()

    consume_cnt = ForwarderConsumeCounters.of_path(net, "S", "D")
    assert consume_cnt.consumed_fidelity_values is not None
    return Stats(
        count=consume_cnt.n_consumed,
        fid=consume_cnt.consumed_fidelity_values,
    )


def run_row(args: Args, ri: RowInput) -> Row:
    details: list[Stats] = []
    for i in range(args.runs):
        details.append(run_simulation(SEED_BASE + i, args, ri))
    return Row(f_req=ri.f_req, details=details)


def main(args: Args) -> list[Row]:
    with Pool(processes=args.workers) as pool:
        rows = pool.starmap(run_row, itertools.product([args], ROW_INPUTS))

    if args.json:
        with open(args.json, "w") as file:
            json.dump(rows, file)

    if args.csv:
        with open(args.csv, "w", newline="") as file:
            w = csv.writer(file)
            w.writerow(["f_req", "rate_mean", "rate_std", "f_mean", "f_std"])
            for row in rows:
                rates = np.fromiter((s["count"] for s in row["details"]), dtype=float) / args.sim_duration
                fids = np.fromiter((f for s in row["details"] for f in s["fid"]), dtype=float)
                if len(fids) == 0:
                    fids = np.array([0])
                w.writerow([row["f_req"], rates.mean(), rates.std(), fids.mean(), fids.std()])

    return rows


if __name__ == "__main__":
    freeze_support()
    args = Args().parse_args()
    main(args)
    # TODO plotting
