"""
This script runs a single repeater with active fidelity enforcement via T_wait.
It gathers statistics of the end-to-end rate and fidelity.
"""

import itertools
from collections.abc import Sequence
from multiprocessing import Pool, freeze_support
from typing import Any, TypedDict, cast, override

import numpy as np
import pandas as pd
from tap import Tap

from mqns.network.builder import CTRL_DELAY, EprTypeLiteral, LinkArchLiteral, NetworkBuilder, tap_configure
from mqns.network.proactive import CutoffSchemeWaitTime, ProactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import log, rng

from examples_common.plotting import Axes1D, SubFigure1D, plt, plt_save

log.set_default_level("CRITICAL")


class Args(Tap):
    workers: int = 1  # number of workers for parallel execution
    runs: int = 100  # number of trials per parameter set
    sim_duration: float = 5.0  # simulation duration in seconds
    L: tuple[float, float] = (32, 18)  # qchannel lengths (km)
    t_cohere: list[float] = [0.1]  # memory coherence time (s)
    t_wait: list[float] = [0.0025, 0.005, 0.01, 0.02, 1000]  # wait-time cutoff values (s)
    epr_type: EprTypeLiteral  # network-wide EPR type
    link_arch: LinkArchLiteral  # link architecture
    link_arch_sim: bool = False  # determine fidelity with LinkArch mini simulation
    fiber_error: str = "DEPOLAR:0.01"  # fiber error model with decoherence rate
    csv: str = ""  # save results as CSV file
    plt: str = ""  # save plot as image file

    @override
    def configure(self) -> None:
        super().configure()
        tap_configure(self)


SIMULATOR_ACCURACY = 1000000
SEED_BASE = 100


def run_simulation(seed: int, args: Args, t_cohere: float, t_wait: float):
    rng.reseed(seed)

    net = (
        NetworkBuilder(
            epr_type=args.epr_type,
        )
        .topo_linear(
            nodes=("S", "R", "D"),
            t_cohere=t_cohere,
            channel_length=args.L,
            fiber_error=args.fiber_error,
            link_arch=args.link_arch,
            init_fidelity=None if args.link_arch_sim else 0.99,
        )
        .proactive_centralized()
        .path(
            "S-D",
            swap=[1, 0, 1],
            swap_cutoff=[0, t_wait, 0],
        )
        .make_network()
    )

    fwS = net.get_node("S").get_app(ProactiveForwarder)
    fwS.cnt.enable_collect_all()
    fwR = net.get_node("R").get_app(ProactiveForwarder)
    waitR = CutoffSchemeWaitTime.of(fwR)
    waitR.cnt.enable_collect_all()

    s = Simulator(0, args.sim_duration + CTRL_DELAY, accuracy=SIMULATOR_ACCURACY, install_to=(log, net))
    s.run()

    rate = fwS.cnt.n_consumed / args.sim_duration
    discard = fwR.cnt.n_cutoff[0] / args.sim_duration
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


FMIN_THRESHOLDS = np.linspace(50, 100)


class Details(TypedDict):
    t_cohere: float
    t_wait: float
    rate_hist: tuple[np.ndarray, np.ndarray]  # rate histogram
    discard_hist: tuple[np.ndarray, np.ndarray]  # discard histogram
    fid_hist: tuple[np.ndarray, np.ndarray]  # fidelity histogram
    wait_hist: tuple[np.ndarray, np.ndarray]  # wait-time histogram
    fid_min: float  # minimum fidelity
    fmin_rate: Sequence[float]  # FMIN_THRESHOLDS - rate curve


def run_row(args: Args, t_cohere: float, t_wait: float) -> tuple[Stats, Details]:
    columns: list[list[float]] = [[] for _ in range(4)]
    for i in range(args.runs):
        print(f"T_cohere={t_cohere:.4f}, T_wait={t_wait:.4f}, run {i + 1}")
        results = run_simulation(SEED_BASE + i, args, t_cohere, t_wait)
        for col, res in zip(columns, results, strict=True):
            col.extend(res)

    rate, discard, fid, wait = columns
    fid = np.multiply(fid, 1e2)
    wait = np.multiply(wait, 1e3 / SIMULATOR_ACCURACY)

    total_time = args.runs * args.sim_duration
    fmin_rates = [np.sum(fid >= t) / total_time for t in FMIN_THRESHOLDS]

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
    ), Details(
        t_cohere=t_cohere,
        t_wait=t_wait,
        rate_hist=np.histogram(rate, bins=16),
        discard_hist=np.histogram(discard, bins=16),
        fid_hist=np.histogram(fid, bins=16),
        wait_hist=np.histogram(wait, bins=16),
        fid_min=np.min(fid),
        fmin_rate=fmin_rates,
    )


HISTOGRAM_INFO = [
    ("fid", "Fidelity (%)", "left"),
    ("wait", "wait-time (ms)", "right"),
]


def plot(rows: list[tuple[Stats, Details]], *, save_plt: str):
    unit_width, unit_height = 2.5, 2.5
    fig = plt.figure(figsize=(unit_width * 4, unit_height * (1 + len(rows))))
    fig.tight_layout()

    subfigs = cast(SubFigure1D, fig.subfigures(nrows=1 + len(rows), ncols=1, hspace=0.2))
    for subfig in subfigs:
        subfig.subplots_adjust(bottom=0.2)

    subfig = subfigs[0]
    subfig.suptitle("Rate vs. Minimum Fidelity Threshold")
    ax = subfig.subplots(nrows=1, ncols=1)
    for stats, details in rows:
        fmin_rates = details["fmin_rate"]
        ax.plot(FMIN_THRESHOLDS, fmin_rates, label=f"T_wait={stats['t_wait']:.4f}", linewidth=1.5)
    ax.set_xbound(max(50, min(details["fid_min"] for _, details in rows)), 100)
    ax.set_xlabel("Minimum Fidelity Threshold $F_{min}$ (%)")
    ax.set_ylabel("Rate (Hz)")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)

    last_axs: Axes1D = []
    for subfig, (stats, details) in zip(subfigs[1:], rows, strict=True):
        subfig.suptitle(
            f"T_cohere={stats['t_cohere']:.4f} "
            f"T_wait={stats['t_wait']:.4f} "
            f"rate={stats['rate_mean']:.2f}\xb1{stats['rate_std']:.2f} "
            f"discard={stats['discard_mean']:.2f}\xb1{stats['discard_std']:.2f} ",
        )
        axs = cast(Axes1D, subfig.subplots(nrows=1, ncols=2))
        last_axs = axs
        for ax, (field, _, label_loc) in zip(axs, HISTOGRAM_INFO, strict=True):
            counts, bins = cast(tuple[np.ndarray, np.ndarray], details[f"{field}_hist"])
            ax.bar(bins[:-1], counts, width=np.diff(bins), color="coral", edgecolor="black", align="edge")
            ax.set_title(f" {stats[f'{field}_mean']:.3f}\xb1{stats[f'{field}_std']:.3f} ", y=0.8, loc=cast(Any, label_loc))
            ax.set_ylabel("counts")

    for ax, (_, title, _) in zip(last_axs, HISTOGRAM_INFO, strict=True):
        ax.set_xlabel(title)

    plt_save(save_plt)


if __name__ == "__main__":
    freeze_support()
    args = Args().parse_args()

    with Pool(processes=args.workers) as pool:
        rows = pool.starmap(run_row, itertools.product([args], args.t_cohere, args.t_wait))

    if args.csv:
        df = pd.DataFrame([s for s, _ in rows])
        df.to_csv(args.csv, index=False)

    plot(rows, save_plt=args.plt)
