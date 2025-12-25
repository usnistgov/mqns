import json
import os.path

import numpy as np
import pandas as pd
from tap import Tap

from examples_common.plotting import plt, plt_save

"""
See scalability_randomtopo.sh for how to run this script.
"""


# Command line arguments
class Args(Tap):
    indir: str  # input directory for MQNS results
    indir_sequence: str = ""  # input directory for SeQUeNCe results
    runs: int = 1  # number of simulation runs per parameter set
    qchannel_capacity: int = 100  # quantum channel capacity
    time_limit: float = 10800.0  # wall-clock limit in seconds
    csv: str = ""  # save results as CSV file
    plt: str = ""  # save plot as image file


args = Args().parse_args()

network_sizes: list[tuple[int, int]] = [
    (16, 20),
    (32, 40),
    (64, 80),
    (128, 160),
    (256, 320),
    (512, 640),
]

SEED_BASE = 200

# Load intermediate files saved by scalability_randomtopo_run.py.
n_simulators = 0
rows: list[dict] = []
for indir, simulator_name in (args.indir, "MQNS"), (args.indir_sequence, "SeQUeNCe"):
    if indir == "":
        continue
    n_simulators += 1
    for nnodes, nedges in network_sizes:
        values = np.zeros(args.runs, dtype=np.float64)
        for j in range(args.runs):
            seed = SEED_BASE + j
            filename = f"{args.qchannel_capacity}-{nnodes}-{nedges}-{seed}.json"
            with open(os.path.join(indir, filename), "r") as file:
                data1 = dict(json.load(file))
                values[j] = data1["time_spent"] / data1.get("sim_progress", 1.0)
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

df = pd.DataFrame(rows)

x_ticks = list(range(len(network_sizes)))
x_ticklabels = [f"({n},{e})" for (n, e) in network_sizes]

# Save CSV output, if requested.
if args.csv:
    df.to_csv(args.csv, index=False)

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
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for simulator_name, data in df.groupby("simulator"):
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
