"""
This script runs a single repeater with active fidelity enforcement via T_wait.
It gathers statistics of the end-to-end rate and fidelity.
"""

import itertools
import json
from collections.abc import Sequence
from multiprocessing import Pool, freeze_support
from typing import Any, TypedDict, cast, override

import numpy as np
import pandas as pd
from tap import Tap

from mqns.entity.base_channel import default_light_speed
from mqns.network.builder import CTRL_DELAY, EprTypeLiteral, LinkArchLiteral, NetworkBuilder, tap_configure
from mqns.network.proactive import CutoffSchemeWaitTime, ProactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import json_default, log, rng

from examples_common.plotting import Axes, Axes1D, SubFigure, SubFigure1D, plt, plt_save

log.set_default_level("CRITICAL")


class Args(Tap):
    workers: int = 1  # number of workers for parallel execution
    runs: int = 100  # number of trials per parameter set
    sim_duration: float = 5.0  # simulation duration in seconds
    L: tuple[float, float] = (32, 18)  # qchannel lengths (km)
    t_cohere: list[float] = [0.1]  # memory coherence time (s)
    t_wait: list[float] = [0.0025, 0.005, 0.01, 0.02, 1000]  # wait-time cutoff values (s), empty for auto-sweep
    epr_type: EprTypeLiteral  # network-wide EPR type
    link_arch: LinkArchLiteral  # link architecture
    link_arch_sim: bool = False  # determine fidelity with LinkArch mini simulation
    fiber_error: str = "DEPOLAR:0.01"  # fiber error model with decoherence rate
    csv: str = ""  # save stats as CSV file
    json: str = ""  # save stats and details as JSON file
    plt: str = ""  # save plot as image file
    plt_from_json: str = ""  # skip simulation and only plot from saved JSON

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
    log.install(None)

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


HISTOGRAM_BINS = 32
FMIN_THRESHOLDS = np.linspace(50, 100)
type HistogramData = tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float]]


class Details(TypedDict):
    t_cohere: float
    t_wait: float
    rate_hist: HistogramData  # rate histogram
    discard_hist: HistogramData  # discard histogram
    fid_hist: HistogramData  # fidelity histogram
    wait_hist: HistogramData  # wait-time histogram
    fid_min: float  # minimum fidelity
    fid_max: float  # maximum fidelity
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
        rate_hist=np.histogram(rate, bins=HISTOGRAM_BINS),
        discard_hist=np.histogram(discard, bins=HISTOGRAM_BINS),
        fid_hist=np.histogram(fid, bins=HISTOGRAM_BINS),
        wait_hist=np.histogram(wait, bins=HISTOGRAM_BINS),
        fid_min=np.min(fid),
        fid_max=np.max(fid),
        fmin_rate=fmin_rates,
    )


HISTOGRAM_INFO = [
    ("fid", "Fidelity (%)", "left"),
    ("wait", "wait-time (ms)", "right"),
]

type Rows = list[tuple[Stats, Details]]


def _ms(n: float) -> str:
    return f"{n * 1000}ms"


def _plot_wait_rate(ax: Axes, rows: Rows):
    ax.errorbar(
        range(len(rows)),
        [stats["rate_mean"] for stats, _ in rows],
        yerr=[stats["rate_std"] for stats, _ in rows],
        fmt="o",
        color="orange",
        ecolor="orange",
        capsize=4,
        label="sim.",
        linestyle="--",
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{stats['t_wait'] * 1000}" for stats, _ in rows])
    ax.set_xlabel("wait-time (ms)")
    ax.set_ylabel("Rate (Hz)")
    ax.grid(True, linestyle="--", alpha=0.6)


def _plot_fmin_rate(ax: Axes, rows: Rows):
    for stats, details in rows:
        fmin_rates = details["fmin_rate"]
        ax.plot(FMIN_THRESHOLDS, fmin_rates, label=f"T_wait={_ms(stats['t_wait'])}", linewidth=1.5)
    ax.set_xbound(
        max(50, min(details["fid_min"] for _, details in rows)),
        min(100, max(details["fid_max"] for _, details in rows) + 2),
    )
    ax.set_xlabel("Minimum Fidelity Threshold $F_{min}$ (%)")
    ax.set_ylabel("Rate (Hz)")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)


def _plot_row(subfig: SubFigure, stats: Stats, details: Details) -> Axes1D:
    subfig.suptitle(
        f"T_cohere={_ms(stats['t_cohere'])} "
        f"T_wait={_ms(stats['t_wait'])} "
        f"rate={stats['rate_mean']:.2f}\xb1{stats['rate_std']:.2f}Hz "
        f"discard={stats['discard_mean']:.2f}\xb1{stats['discard_std']:.2f}Hz ",
    )
    axs = cast(Axes1D, subfig.subplots(nrows=1, ncols=2))
    for ax, (field, _, label_loc) in zip(axs, HISTOGRAM_INFO, strict=True):
        counts, bins = cast(tuple[np.ndarray, np.ndarray], details[f"{field}_hist"])
        ax.bar(bins[:-1], counts, width=np.diff(bins), color="coral", edgecolor="black", align="edge")
        ax.set_title(f" {stats[f'{field}_mean']:.3f}\xb1{stats[f'{field}_std']:.3f} ", y=0.8, loc=cast(Any, label_loc))
        ax.set_ylabel("counts")
    return axs


def plot(rows: Rows, *, save_plt: str):
    unit_width, unit_height = 2.5, 2.5
    fig = plt.figure(figsize=(unit_width * 4, unit_height * (1 + len(rows))))
    fig.tight_layout()

    subfigs = cast(SubFigure1D, fig.subfigures(nrows=1 + len(rows), ncols=1, hspace=0.2))
    for subfig in subfigs:
        subfig.subplots_adjust(bottom=0.2)

    subfig = subfigs[0]
    axs = cast(Axes1D, subfig.subplots(nrows=1, ncols=2))
    _plot_wait_rate(axs[0], rows)
    _plot_fmin_rate(axs[1], rows)

    last_axs: Axes1D = []
    for subfig, (stats, details) in zip(subfigs[1:], rows, strict=True):
        last_axs = _plot_row(subfig, stats, details)

    for ax, (_, title, _) in zip(last_axs, HISTOGRAM_INFO, strict=True):
        ax.set_xlabel(title)

    plt_save(save_plt)


def main(args: Args) -> Rows:
    t_wait: Sequence[float] = args.t_wait
    rows: Rows = []
    if len(args.t_wait) == 0:
        if len(args.t_cohere) != 1:
            raise ValueError("expect exactly one t_cohere for t_wait auto-sweep")
        rows.append(row0 := run_row(args, args.t_cohere[0], args.t_cohere[0]))
        stats0_wait_mean, stats0_wait_std = row0[0]["wait_mean"], row0[0]["wait_std"]
        log.info(f"t_wait auto-sweep unrestricted run wait={stats0_wait_mean}\xb1{stats0_wait_std}")
        t_wait_max = min(args.t_cohere[0] / 2, stats0_wait_mean + stats0_wait_std)
        t_wait_min = max(args.L) / default_light_speed[0]
        t_wait_cur, t_wait = t_wait_min, []
        while True:
            t_wait.append(t_wait_cur)
            if t_wait_cur >= t_wait_max:
                break
            t_wait_cur *= 2
        log.info(f"t_wait auto-sweep {t_wait_min}..{t_wait_max} => {t_wait}")

    with Pool(processes=args.workers) as pool:
        rows.extend(pool.starmap(run_row, itertools.product([args], args.t_cohere, t_wait)))

    rows.sort(key=lambda row: (row[0]["t_cohere"], row[0]["t_wait"]))

    if args.csv:
        df = pd.DataFrame([s for s, _ in rows])
        df.to_csv(args.csv, index=False)

    if args.json:
        with open(args.json, "w") as file:
            json.dump(rows, file, default=json_default)

    return rows


if __name__ == "__main__":
    freeze_support()
    args = Args().parse_args()

    if args.plt_from_json:
        with open(args.plt_from_json, "r") as file:
            rows: Rows = json.load(file)
    else:
        rows = main(args)

    plot(rows, save_plt=args.plt)
