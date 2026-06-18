"""
Simulate a 3-node linear topology and report end-to-end throughput.

.. figure:: /_static/examples/3_nodes_32_18.svg
   :alt: 3-node linear topology
   :align: center

This script sets up and executes parallelized quantum network simulations using:

* A linear network with configurable quantum channel lengths (``--L``).
* Heterogeneous or uniform memory allocation patterns per channel direction (``--M``).
* Network-wide choices for EPR pair types (``--epr_type``) and link architectures (``--link_arch``).
* Operation modes (``--mode``):
    * ``PCA``: Proactive forwarding, Centralized control, Async timing.
    * ``PCS``: Proactive forwarding, Centralized control, Sync timing.
    * ``PCA``: Reactive forwarding, Centralized control, Sync timing.
* User-provided SYNC timing phase durations via ``--sync_timing``.
* A seeded random number generator running across parallel execution pool workers.

After simulation execution, it aggregates and reports end-to-end entanglement generation rates,
mean quantum state fidelities, and memory decoherence metrics over a sweep of qubit coherence times.
The resulting statistics can be saved to JSON/CSV files and plotted.
"""

import itertools
import json
from multiprocessing import Pool, freeze_support
from typing import Literal, TypedDict, override

import numpy as np
import pandas as pd
from tap import Tap

from mqns.network.builder import CTRL_DELAY, ChannelParam, EprTypeLiteral, LinkArchLiteral, NetworkBuilder, tap_configure
from mqns.network.fw import ForwarderConsumeCounters
from mqns.network.protocol.link_layer import LinkLayerCounters
from mqns.simulator import Simulator
from mqns.utils import log, rng

from examples_common.plotting import plt, plt_save

log.set_default_level("CRITICAL")


class Args(Tap):
    workers: int = 1  # number of workers for parallel execution
    runs: int = 100  # number of trials per parameter set
    sim_duration: float = 3  # simulation duration in seconds
    mode: Literal["PCA", "PCS", "RCS"] = "PCA"
    sync_timing: list[float]
    L: tuple[float, float] = (32, 18)  # qchannel lengths (km)
    M: list[int] = [1]
    t_cohere: list[float] = [0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1]  # memory coherence times (sec)
    epr_type: EprTypeLiteral  # network-wide EPR type
    link_arch: LinkArchLiteral  # link architecture
    json: str = ""  # save results as JSON file
    csv: str = ""  # save summary as CSV file
    plt: str = ""  # save plot as image file

    @override
    def configure(self) -> None:
        tap_configure(self)

        self.add_argument("--L", metavar=("L_SR", "L_RD"))
        self.add_argument("--M", help="(int | 4*int, default: 1) number of qubits per channel per direction")

    @override
    def process_args(self) -> None:
        if len(self.M) not in (1, 4):
            raise ValueError("--M must have either 1 or 4 elements")
        if min(self.M) < 1:
            raise ValueError("--M must be positive")

    def linear_channels(self) -> tuple[ChannelParam, ChannelParam]:
        """
        Derive ``NetworkBuilder.topo_linear(channels=)`` from ``--L`` and ``--M``.

        If ``--M`` has one integer, it is used as channel capacity on both S-R and R-D.

        If ``--M`` has four integers, they are used as:

        1. number of qubits on S assigned to S-R channel
        2. number of qubits on R assigned to S-R channel
        3. number of qubits on R assigned to R-D channel
        4. number of qubits on D assigned to R-D channel
        """
        lSR, lRD = self.L
        if len(self.M) == 1:
            mSr = mRl = mRr = mDl = self.M[0]
        else:
            mSr, mRl, mRr, mDl = self.M
        return (
            ChannelParam(ch_length=lSR, ch_capacity=(mSr, mRl)),
            ChannelParam(ch_length=lRD, ch_capacity=(mRr, mDl)),
        )


SEED_BASE = 100


class Stats(TypedDict):
    t_cohere: float
    throughput_eps: float
    mean_fidelity: float
    expired_ratio: float
    expired_per_e2e: float


def run_simulation(seed: int, args: Args, t_cohere: float) -> Stats:
    rng.reseed(seed)

    b = NetworkBuilder().topo_linear(
        nodes="SRD",
        t_cohere=t_cohere,
        channels=args.linear_channels(),
        link_arch=args.link_arch,
    )

    total_duration = args.sim_duration
    match args.mode:
        case "PCA":
            total_duration += CTRL_DELAY
            b.proactive_centralized()
        case "PCS":
            b.proactive_centralized(timing=args.sync_timing)
        case "RCS":
            b.reactive_centralized(timing=args.sync_timing)

    b.request("S-D")
    net = b.make_network()
    del b

    s = Simulator(0, total_duration, accuracy=1000000, install_to=(log, net))
    s.run()

    consume_cnt = ForwarderConsumeCounters.of_path(net, "S", "D")
    ll_cnt = LinkLayerCounters.aggregate(net.nodes)
    stats = Stats(
        t_cohere=t_cohere,
        throughput_eps=consume_cnt.get_rate(args.sim_duration),
        mean_fidelity=consume_cnt.consumed_avg_fidelity,
        expired_ratio=ll_cnt.decoh_ratio,
        expired_per_e2e=consume_cnt.get_per_consumed(ll_cnt.n_decoh),
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
