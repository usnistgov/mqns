"""
This script runs a single repeater with active fidelity enforcement via T_wait.
It gathers statistics of the end-to-end rate and fidelity.
"""

import itertools
from multiprocessing import Pool, freeze_support
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
from tap import Tap

from mqns.network.network import QuantumNetwork
from mqns.network.proactive import CutoffSchemeWaitTime, ProactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from examples_common.plotting import Axes1D, SubFigure1D, plt, plt_save
from examples_common.topo_linear import build_topology

log.set_default_level("CRITICAL")


class Args(Tap):
    workers: int = 1  # number of workers for parallel execution
    runs: int = 100  # number of trials per parameter set
    csv: str = ""  # save results as CSV file
    plt: str = ""  # save plot as image file


SIMULATOR_ACCURACY = 1000000
SEED_BASE = 100
sim_duration = 5.0


def run_simulation(seed: int, t_cohere: float, t_wait: float):
    set_seed(seed)

    topo = build_topology(
        nodes=["S", "R", "D"],
        t_cohere=t_cohere,
        channel_length=[32, 18],
        swap=[1, 0, 1],
        swap_cutoff=[0, t_wait, 0],
    )
    net = QuantumNetwork(topo)

    fwS = net.get_node("S").get_app(ProactiveForwarder)
    fwS.cnt.enable_collect_all()
    fwR = net.get_node("R").get_app(ProactiveForwarder)
    waitR = CutoffSchemeWaitTime.of(fwR)
    waitR.cnt.enable_collect_all()

    s = Simulator(0, sim_duration + 5e-06, accuracy=SIMULATOR_ACCURACY, install_to=(log, net))
    s.run()

    rate = fwS.cnt.n_consumed / sim_duration
    discard = fwR.cnt.n_cutoff[0] / sim_duration
    assert fwS.cnt.consumed_fidelity_values is not None
    assert waitR.cnt.wait_values is not None
    return [rate], [discard], fwS.cnt.consumed_fidelity_values, waitR.cnt.wait_values


class Stats(TypedDict):
    t_cohere: float  # input memory coherence time
    t_wait: float  # input wait-time budget
    rate_mean: float  # end-to-end entanglements per second
    rate_std: float
    discard_mean: float  # discarded elementary entanglements per second
    discard_std: float
    fid_mean: float  # end-to-end fidelity among all end-to-end entanglements (%)
    fid_std: float
    wait_mean: float  # actual wait-time among all waited elementary entanglements (milliseconds)
    wait_std: float


class Histograms(TypedDict):
    t_cohere: float
    t_wait: float
    rate: tuple[np.ndarray, np.ndarray]
    discard: tuple[np.ndarray, np.ndarray]
    fid: tuple[np.ndarray, np.ndarray]
    wait: tuple[np.ndarray, np.ndarray]


def run_row(n_runs: int, t_cohere: float, t_wait: float) -> tuple[Stats, Histograms]:
    columns: list[list[float]] = [[] for _ in range(4)]
    for i in range(n_runs):
        print(f"T_cohere={t_cohere:.4f}, T_wait={t_wait:.4f}, run {i + 1}")
        results = run_simulation(SEED_BASE + i, t_cohere, t_wait)
        for col, res in zip(columns, results, strict=True):
            col.extend(res)

    rate, discard, fid, wait = columns
    fid = np.multiply(fid, 1e2)
    wait = np.multiply(wait, 1e3 / SIMULATOR_ACCURACY)

    return Stats(
        t_cohere=t_cohere,
        t_wait=t_wait,
        rate_mean=np.mean(rate).item(),
        rate_std=np.std(rate).item(),
        discard_mean=np.mean(discard).item(),
        discard_std=np.std(discard).item(),
        fid_mean=np.mean(fid).item(),
        fid_std=np.std(fid).item(),
        wait_mean=np.mean(wait).item(),
        wait_std=np.std(wait).item(),
    ), Histograms(
        t_cohere=t_cohere,
        t_wait=t_wait,
        rate=np.histogram(rate, bins=16),
        discard=np.histogram(discard, bins=16),
        fid=np.histogram(fid, bins=16),
        wait=np.histogram(wait, bins=16),
    )


HISTOGRAM_INFO = [
    ("fid", "Fidelity (%)", "left"),
    ("wait", "wait-time (ms)", "right"),
]


def plot(rows: list[tuple[Stats, Histograms]], *, save_plt: str):
    unit_width, unit_height = 2.5, 2.5
    fig = plt.figure(figsize=(unit_width * 4, unit_height * len(rows)))
    fig.tight_layout()

    subfigs = cast(SubFigure1D, fig.subfigures(nrows=len(rows), ncols=1, hspace=0.1))
    last_axs: Axes1D = []
    for subfig, (stats, histograms) in zip(subfigs, rows, strict=True):
        subfig.suptitle(
            f"T_cohere={stats['t_cohere']:.4f} "
            f"T_wait={stats['t_wait']:.4f} "
            f"rate={stats['rate_mean']:.2f}\xb1{stats['rate_std']:.2f} "
            f"discard={stats['discard_mean']:.2f}\xb1{stats['discard_std']:.2f} "
        )

        axs = cast(Axes1D, subfig.subplots(nrows=1, ncols=2))
        subfig.subplots_adjust(wspace=0.2, bottom=0.2)
        last_axs = axs
        for ax, (field, _, label_loc) in zip(axs, HISTOGRAM_INFO, strict=True):
            counts, bins = cast(tuple[np.ndarray, np.ndarray], histograms[field])
            ax.bar(bins[:-1], counts, width=np.diff(bins), color="coral", edgecolor="black", align="edge")
            ax.set_title(f" {stats[f'{field}_mean']:.3f}\xb1{stats[f'{field}_std']:.3f} ", y=0.8, loc=cast(Any, label_loc))
            ax.set_ylabel("counts")

    for ax, (_, title, _) in zip(last_axs, HISTOGRAM_INFO, strict=True):
        ax.set_xlabel(title)

    plt_save(save_plt)


t_cohere_values = [0.1]
t_wait_values = [0.0025, 0.005, 0.01, 0.02, 1000]


if __name__ == "__main__":
    freeze_support()
    args = Args().parse_args()

    with Pool(processes=args.workers) as pool:
        rows = pool.starmap(run_row, itertools.product([args.runs], t_cohere_values, t_wait_values))

    if args.csv:
        df = pd.DataFrame([s for s, _ in rows])
        df.to_csv(args.csv, index=False)

    plot(rows, save_plt=args.plt)
