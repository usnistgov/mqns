"""
Simulate 3-node linear topology and report end-to-end throughput.

This script sets up and executes simulations using:

* A generated topology based with varying qubit coherence time,
* A quantum network with Dijkstra-based routing algorithm, and asynchronous timing mode,
* A seeded random number generator.

After simulation, it reports the number of successful entanglement generations per second.
The statistics can be saved and plotted.
"""

import itertools
import json
from multiprocessing import Pool, freeze_support
from typing import Literal, TypedDict, override

import numpy as np
import pandas as pd
from tap import Tap

from mqns.network.builder import CTRL_DELAY, EprTypeLiteral, LinkArchLiteral, NetworkBuilder, tap_configure
from mqns.network.fw import Forwarder
from mqns.network.network import TimingModeSync
from mqns.network.protocol.link_layer import LinkLayerCounters
from mqns.simulator import Simulator
from mqns.utils import log, rng

from examples_common.plotting import plt, plt_save

log.set_default_level("CRITICAL")


class Args(Tap):
    workers: int = 1  # number of workers for parallel execution
    runs: int = 100  # number of trials per parameter set
    sim_duration: float = 3  # simulation duration in seconds
    mode: Literal["P", "R"] = "P"  # choose proactive or reactive mode
    L: tuple[float, float] = (32, 18)  # qchannel lengths (km)
    t_cohere: list[float] = [0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1]  # memory coherence time (s)
    qchannel_capacity: int = 1  # qchannel capacity
    epr_type: EprTypeLiteral  # network-wide EPR type
    link_arch: LinkArchLiteral  # link architecture
    json: str = ""  # save results as JSON file
    csv: str = ""  # save results as CSV file
    plt: str = ""  # save plot as image file

    @override
    def configure(self) -> None:
        super().configure()
        tap_configure(self)


SEED_BASE = 100


class Stats(TypedDict):
    t_cohere: float
    throughput_eps: float
    mean_fidelity: float
    expired_ratio: float
    expired_per_e2e: float


def run_simulation(seed: int, args: Args, t_cohere: float) -> Stats:
    rng.reseed(seed)

    match args.mode:
        case "P":
            b = NetworkBuilder()
            total_duration = args.sim_duration + CTRL_DELAY
        case "R":
            b = NetworkBuilder(timing=TimingModeSync(t_ext=0.03, t_rtg=0.00005, t_int=0.0002))
            total_duration = args.sim_duration

    b.topo_linear(
        nodes=("S", "R", "D"),
        t_cohere=t_cohere,
        channel_length=args.L,
        channel_capacity=args.qchannel_capacity,
        link_arch=args.link_arch,
    )

    match args.mode:
        case "P":
            b.proactive_centralized().path("S-D")
        case "R":
            b.reactive_centralized()

    net = b.make_network()
    del b

    s = Simulator(0, total_duration, accuracy=1000000, install_to=(log, net))
    s.run()

    fw_s_cnt = net.get_node("S").get_app(Forwarder).cnt
    ll_cnt = LinkLayerCounters.aggregate(net.nodes)
    stats = Stats(
        t_cohere=t_cohere,
        throughput_eps=fw_s_cnt.n_consumed / args.sim_duration,
        mean_fidelity=float(fw_s_cnt.consumed_avg_fidelity),
        expired_ratio=ll_cnt.decoh_ratio,
        expired_per_e2e=ll_cnt.n_decoh / fw_s_cnt.n_consumed if fw_s_cnt.n_consumed > 0 else 0,
    )
    return stats


def run_row(args: Args, t_cohere: float) -> list[Stats]:
    results: list[Stats] = []
    for i in range(args.runs):
        print(f"T_cohere={t_cohere:.4f}, run {i + 1}")
        stats = run_simulation(SEED_BASE + i, args, t_cohere)
        results.append(stats)
    return results


def plot(df: pd.DataFrame, *, save_plt: str):
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        df["T_cohere"],
        df["Mean Rate"],
        yerr=df["Std Rate"],
        fmt="o",
        color="orange",
        ecolor="orange",
        capsize=4,
        label="sim.",
        linestyle="--",
    )
    plt.xscale("log")
    plt.xlabel(r"$T_{\mathrm{cohere}}$")
    plt.ylabel("Ent. per second")
    plt.title("E2e rate")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt_save(save_plt)


if __name__ == "__main__":
    freeze_support()
    args = Args().parse_args()

    with Pool(processes=args.workers) as pool:
        rows = pool.starmap(run_row, itertools.product([args], args.t_cohere))

    if args.json:
        with open(args.json, "w") as file:
            json.dump(rows, file)

    results = {"T_cohere": [], "Mean Rate": [], "Std Rate": []}
    for row in rows:
        rates = [s["throughput_eps"] for s in row]
        results["T_cohere"].append(row[0]["t_cohere"])
        results["Mean Rate"].append(np.mean(rates))
        results["Std Rate"].append(np.std(rates))

    # Final results summary print
    print("\nT_coh    Rate")
    for t, mean, std in zip(results["T_cohere"], results["Mean Rate"], results["Std Rate"]):
        print(f"{t:<7.3f}  {mean:>5.1f} ({std:.1f})")

    df = pd.DataFrame(results)
    if args.csv:
        df.to_csv(args.csv, index=False)

    plot(df, save_plt=args.plt)
