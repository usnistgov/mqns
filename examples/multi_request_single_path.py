import json
from typing import NamedTuple, cast

import numpy as np
from tap import Tap

from mqns.network.builder import CTRL_DELAY, NetworkBuilder
from mqns.network.fw import MuxScheme, MuxSchemeDynamicEpr, MuxSchemeStatistical
from mqns.network.proactive import ProactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import log, rng

from examples_common.plotting import Axes2D, mpl, plt, plt_save

log.set_default_level("CRITICAL")


class Args(Tap):
    runs: int = 3  # number of trials per parameter set
    sim_duration: float = 3  # simulation duration in seconds
    json: str = ""  # save results as JSON file
    plt: str = ""  # save plot as image file


SEED_BASE = 100
PATH_TITLES = ("S1-D1", "S2-D2")
N_PATHS = len(PATH_TITLES)

# Quantum channel lengths
ch_S1_R1 = 10
ch_R1_R2 = 10
ch_R2_R3 = 10
ch_R3_D1 = 10
ch_S2_R2 = 10
ch_R3_D2 = 10


def run_simulation(seed: int, args: Args, mux: MuxScheme, t_cohere: float):
    rng.reseed(seed)

    net = (
        NetworkBuilder()
        .topo(
            channels=[
                ("S1-R1", ch_S1_R1),
                ("R1-R2", ch_R1_R2),
                ("R2-R3", ch_R2_R3),
                ("R3-D1", ch_R3_D1),
                ("S2-R2", ch_S2_R2),
                ("R3-D2", ch_R3_D2),
            ],
            t_cohere=t_cohere,
        )
        .proactive_centralized(mux=mux)
        .path("S1-D1")
        .path("S2-D2")
        .make_network()
    )

    s = Simulator(0, args.sim_duration + CTRL_DELAY, accuracy=1000000, install_to=(log, net))
    s.run()

    #### get stats: e2e_rate and mean_fidelity
    fw_s1 = net.get_node("S1").get_app(ProactiveForwarder)
    fw_s2 = net.get_node("S2").get_app(ProactiveForwarder)
    # [(path 1), (path 2), ...]
    return [
        (fw_s1.cnt.n_consumed / args.sim_duration, fw_s1.cnt.consumed_avg_fidelity),
        (fw_s2.cnt.n_consumed / args.sim_duration, fw_s2.cnt.consumed_avg_fidelity),
    ]


class PathStats(NamedTuple):
    rate_mean: float
    rate_std: float
    fid_mean: float
    fid_std: float


def run_row(args: Args, strategy: str, t_cohere: float) -> list[PathStats]:
    mux = STRATEGIES[strategy]

    path_rates: list[list[float]] = [[] for _ in range(N_PATHS)]
    path_fids: list[list[float]] = [[] for _ in range(N_PATHS)]

    for i in range(args.runs):
        print(f"{strategy}, T_cohere={t_cohere:.3f}, run #{i}")
        res = run_simulation(SEED_BASE + i, args, mux, t_cohere)
        for path, (rate, fid) in enumerate(res):
            path_rates[path].append(rate)
            path_fids[path].append(fid)

    return [
        PathStats(np.mean(rates).item(), np.std(rates).item(), np.mean(fids).item(), np.std(fids).item())
        for rates, fids in zip(path_rates, path_fids, strict=True)
    ]


def plot(results: dict[str, list[list[PathStats]]], *, save_plt: str):
    mpl.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.titlesize": 22,
            "lines.linewidth": 2,
            "lines.markersize": 7,
            "errorbar.capsize": 4,
        }
    )

    fig, axs = plt.subplots(nrows=2, ncols=N_PATHS, figsize=(9, 8), sharex=True, sharey="row")
    axs = cast(Axes2D, axs)

    for strategy in STRATEGIES:
        for (path, path_title), res_sp in zip(enumerate(PATH_TITLES), results[strategy], strict=True):
            # Plot Entanglement Rate
            ax = axs[0, path]
            ax.errorbar(
                [t * 1e3 for t in T_COHERE_VALUES],
                [r.rate_mean for r in res_sp],
                yerr=[r.rate_std for r in res_sp],
                marker="o",
                label=strategy,
            )
            ax.set_title(path_title)
            ax.set_ylabel("E2E Rate (eps)")
            ax.grid(True)

            # Plot Fidelity
            ax = axs[1, path]
            ax.errorbar(
                [t * 1e3 for t in T_COHERE_VALUES],
                [r.fid_mean for r in res_sp],
                yerr=[r.fid_std for r in res_sp],
                marker="s",
                label=strategy,
            )
            ax.set_title(path_title)
            ax.set_xlabel("T_cohere (ms)")
            ax.set_ylabel("Fidelity")
            ax.grid(True)

    axs[1, -1].legend(title="Strategy", loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt_save(args.plt)


# Simulation constants
STRATEGIES: dict[str, MuxScheme] = {
    "Statistical Mux.": MuxSchemeStatistical(),
    "Random Alloc.": MuxSchemeDynamicEpr(),
    "Swap-weighted Alloc.": MuxSchemeDynamicEpr(select_path=MuxSchemeDynamicEpr.SelectPath_swap_weighted),
}
T_COHERE_VALUES = [5e-3, 10e-3, 20e-3]

if __name__ == "__main__":
    args = Args().parse_args()

    results: dict[str, list[list[PathStats]]] = {}  # strategy->path->t_cohere_index
    for strategy in STRATEGIES:
        results[strategy] = [[] for _ in PATH_TITLES]
        for t_cohere in T_COHERE_VALUES:
            row = run_row(args, strategy, t_cohere)
            for path, stats in enumerate(row):
                results[strategy][path].append(stats)

    if args.json:
        with open(args.json, "w") as file:
            json.dump(results, file)

    plot(results, save_plt=args.plt)
