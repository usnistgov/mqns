"""
The scalability_randomtopo experiment measures how simulation performance and outcomes scale as the network size increases.

* A random topology is used with an average node degree of 2.5.
* For each network size, the number of entanglement requests is chosen to be proportional to the number of nodes,
  with 20% of nodes involved in src-dst requests (plus intermediate nodes).
* Proactive forwarding is used with Statistical multiplexing and SWAP-ASAP swapping policy.
* Each simulation reports execution time, along with other metrics for verification.

Example scripts related to this experiment are:

* ``scalability_randomtopo_run.py``: run the scenario with MQNS.
* ``sequence/scalability_randomtopo_run.py``: run the scenario with SeQUeNCe simulator.
* ``scalability_randomtopo_plot.py``: plot the execution time results.

See ``scalability_randomtopo.sh`` for how to run these scripts.
"""

import json
import os.path

import numpy as np
import pandas as pd
from tap import Tap

from examples_common.plotting import plt, plt_save
from examples_common.scalability_randomtopo import RunResult


class Args(Tap):
    indir: str  # input directory
    sequence: bool = False  # compare with SeQUeNCe
    runs: int = 1  # number of simulation runs per parameter set
    qchannel_capacity: int = 10  # quantum channel capacity
    time_limit: float = 10800.0  # wall-clock limit in seconds
    csv: str = ""  # save results as CSV file
    plt: str = ""  # save plot as image file


network_sizes: list[tuple[int, int]] = [
    (16, 20),
    (32, 40),
    (64, 80),
    (128, 160),
    (256, 320),
    (512, 640),
]

SEED_BASE = 200


def load_results(args: Args) -> pd.DataFrame:
    """Load intermediate files saved by scalability_randomtopo_run.py."""
    rows: list[dict] = []
    for enabled, simulator_name, suffix in (True, "MQNS", ".json"), (args.sequence, "SeQUeNCe", ".sequence.json"):
        if not enabled:
            continue
        for nnodes, nedges in network_sizes:
            values = np.zeros(args.runs, dtype=np.float64)
            for j in range(args.runs):
                seed = SEED_BASE + j
                filename = f"{args.qchannel_capacity}-{nnodes}-{nedges}-{seed}{suffix}"
                with open(os.path.join(args.indir, filename)) as file:
                    data1 = RunResult(json.load(file))
                    values[j] = data1["time_spent"] / data1["sim_progress"]
            rows.append(
                {
                    "simulator": simulator_name,
                    "nnodes": nnodes,
                    "nedges": nedges,
                    "mean": np.mean(values).item(),
                    "std": np.std(values).item(),
                    "values": values,
                }
            )

    return pd.DataFrame(rows)


def plot_results(args: Args, df: pd.DataFrame) -> None:
    x_ticks = list(range(len(network_sizes)))
    x_ticklabels = [f"({n},{e})" for (n, e) in network_sizes]

    # Set plot style.
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "lines.linewidth": 2,
            "lines.markersize": 7,
        }
    )

    # Generate simulation execution time plot.
    _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    n_simulators = 0
    for simulator_name, data in df.groupby("simulator"):
        n_simulators += 1
        data1 = data.sort_values("nnodes")
        assert len(data1) == len(network_sizes)
        ax.errorbar(x_ticks, data1["mean"], yerr=data1["std"], marker="o", linestyle="-", label=simulator_name)

    if df["mean"].max() > args.time_limit:
        ax.axhline(args.time_limit, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xticks(x_ticks, x_ticklabels, rotation=30, ha="right")
    ax.set_xlabel("Network size (#nodes,#edges)")
    ax.set_ylabel("Execution Time (s)")
    if n_simulators > 1:
        ax.legend()
    ax.grid(True, alpha=0.4)

    plt_save(args.plt)


if __name__ == "__main__":
    args = Args().parse_args()
    df = load_results(args)
    if args.csv:
        df.to_csv(args.csv, index=False)
    plot_results(args, df)
