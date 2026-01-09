"""
This script simulates a linear network without swapping and reports per-link statistics.

The network consists of one or more quantum channels.
The quantum channel lengths can be specified with `-L` flags.

One or more memory pairs are assigned to each channel.
The number of memory pairs can be specified with `-M` flags.
If multiple numbers are specified, they will be simulated in separate scenarios.

The following statistics are gathered for each quantum channel:

* Attempts rate: how many entanglements are attempted per second.
* Entanglement rate: how many entanglements are established per second.
* Success rate: Entanglement rate divided by Attempts rate.

The results are saved as JSON (every simulation run), CSV (mean and stdev per scenario), and plots.
"""

import itertools
import json
from multiprocessing import Pool, freeze_support
from typing import TypedDict

import numpy as np
import pandas as pd
from tap import Tap

from mqns.network.network import QuantumNetwork
from mqns.network.protocol.link_layer import LinkLayer
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from examples_common.plotting import Axes, plt, plt_save
from examples_common.topo_linear import build_topology

log.set_default_level("CRITICAL")

SEED_BASE = 100


class ChannelResult(TypedDict):
    L: float
    M: int
    Attempts: float
    Entanglement: float
    Success: float


def run_simulation(seed: int, sim_duration: float, L: list[float], M: int) -> list[ChannelResult]:
    """
    Run single simulation.
    Return per-channel results.
    """
    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = build_topology(
        nodes=len(L) + 1,
        t_cohere=0.1,
        channel_length=L,
        channel_capacity=M,
        swap=[0] * (len(L) + 1),
    )
    net = QuantumNetwork(topo)
    net.install(s)

    s.run()

    res: list[ChannelResult] = []
    for i, length in enumerate(L):
        node = net.get_node("S" if i == 0 else f"R{i}")
        cnt = node.get_app(LinkLayer).cnt
        res.append(
            ChannelResult(
                L=length,
                M=M,
                Attempts=cnt.n_attempts / sim_duration,
                Entanglement=cnt.n_etg / sim_duration,
                Success=0.0 if cnt.n_attempts == 0 else cnt.n_etg / cnt.n_attempts,
            )
        )
    return res


def run_row(n_runs: int, sim_duration: float, L: list[float], M: int) -> list[list[ChannelResult]]:
    """
    Run N simulations.
    Return details from single simulations.
    """
    row: list[list[ChannelResult]] = []
    for i in range(n_runs):
        res = run_simulation(SEED_BASE + i, sim_duration, L, M)
        row.append(res)
    return row


def gather_channel_stats(row: list[list[ChannelResult]], i: int) -> dict:
    ch_res = [res[i] for res in row]
    d = {
        "L": ch_res[0]["L"],
        "M": ch_res[0]["M"],
    }
    for col in ("Attempts", "Entanglement", "Success"):
        a = [res[col] for res in ch_res]
        d[f"{col} rate"] = np.mean(a)
        d[f"{col} std"] = np.std(a)
    return d


def convert_results(table: list[list[list[ChannelResult]]]) -> pd.DataFrame:
    data = {
        "L": [],
        "M": [],
        "Attempts rate": [],
        "Entanglement rate": [],
        "Success rate": [],
        "Attempts std": [],
        "Entanglement std": [],
        "Success std": [],
    }

    for row in table:
        for i_link in range(len(row[0])):
            stats = gather_channel_stats(row, i_link)
            for col, val in stats.items():
                data[col].append(val)

    return pd.DataFrame(data)


SUBPLOT_INFO = [
    ("Attempts", "Attempts rate", "Attempts/s", False),
    ("Entanglement", "Ent. rate", "Ent/s", False),
    ("Success", "Success rate", "Fraction", True),
]


def plot_results(L: list[float], df: pd.DataFrame, *, save_plt: str):
    fig, axs = plt.subplots(1, len(SUBPLOT_INFO), figsize=(10, 4))

    for length in L:
        d = df[df["L"] == length]
        for i, (col, *_) in enumerate(SUBPLOT_INFO):
            ax: Axes = axs[i]
            ax.errorbar(
                d["M"], d[f"{col} rate"], yerr=d[f"{col} std"], label=f"L={length}", marker="o", linestyle="--", capsize=3
            )

    for i, (_, title, ylabel, legend) in enumerate(SUBPLOT_INFO):
        ax: Axes = axs[i]
        ax.set_title(title)
        ax.set_xlabel("M")
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend()

    fig.tight_layout()
    plt_save(save_plt)


if __name__ == "__main__":
    freeze_support()

    # Command line arguments
    class Args(Tap):
        workers: int = 1  # number of workers for parallel execution
        runs: int = 50  # number of trials per parameter set
        sim_duration: float = 1.0  # simulation duration in seconds
        L: list[float] = [32, 18]  # qchannel lengths (km)
        M: list[int] = [1, 2, 3, 4, 5]  # qchannel capacity
        json: str = ""  # save details as JSON file
        csv: str = ""  # save statistics as CSV file
        plt: str = ""  # save plot as image file

    args = Args().parse_args()

    # Simulator loop with process-based parallelism
    with Pool(processes=args.workers) as pool:
        table = pool.starmap(run_row, itertools.product([args.runs], [args.sim_duration], [args.L], args.M))

    if args.json:
        with open(args.json, "w") as file:
            json.dump(table, file)

    df = convert_results(table)
    if args.csv:
        df.to_csv(args.csv, index=False)

    plot_results(args.L, df, save_plt=args.plt)
