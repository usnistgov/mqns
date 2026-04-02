import json
import os.path
import sys

import numpy as np
import pandas as pd

from srt_detail.defs import ParamsArgs, RunResult

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples_common.plotting import plt, plt_save


class Args(ParamsArgs):
    indir: str  # input directory
    sequence: bool = False  # compare with SeQUeNCe
    csv: str = ""  # save results as CSV file
    plt: str = ""  # save plot as image file


def load_results(args: Args) -> pd.DataFrame:
    """Load intermediate files saved by srt_mqns.py and srt_sequence.py."""
    seed_base = args.params["seed_base"]
    runs = args.params["runs"]
    network_sizes = args.params["network_sizes"]

    rows: list[dict] = []
    for enabled, simulator_name, suffix in (True, "MQNS", ".json"), (args.sequence, "SeQUeNCe", ".sequence.json"):
        if not enabled:
            continue
        for ns in network_sizes:
            values = np.zeros(runs, dtype=np.float64)
            for j in range(runs):
                filename = f"{args.qchannel_capacity}-{ns['nodes']}-{ns['edges']}-{seed_base + j}{suffix}"
                with open(os.path.join(args.indir, filename)) as file:
                    data1 = RunResult(json.load(file))
                    values[j] = data1["time_spent"] / data1["sim_progress"]
            rows.append(
                {
                    "simulator": simulator_name,
                    "nodes": ns["nodes"],
                    "edges": ns["edges"],
                    "mean": np.mean(values).item(),
                    "std": np.std(values).item(),
                    "values": values,
                }
            )

    return pd.DataFrame(rows)


def plot_results(args: Args, df: pd.DataFrame) -> None:
    network_sizes = args.params["network_sizes"]

    x_ticks = list(range(len(network_sizes)))
    x_ticklabels = [f"({ns['nodes']},{ns['edges']})" for ns in network_sizes]

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
        data1 = data.sort_values("nodes")
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
