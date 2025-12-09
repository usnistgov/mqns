import json
import os.path

import numpy as np
import pandas as pd
from tap import Tap

from examples_common.plotting import mpl, plt, plt_save

"""
See scalability_randomtopo.sh for how to run this script.
"""


# Command line arguments
class Args(Tap):
    indir: str  # input directory
    runs: int = 1  # number of simulation runs per parameter set
    qchannel_capacity: int = 100  # quantum channel capacity
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
x_ticks = list(range(len(network_sizes)))
x_ticklabels = [f"({n},{e})" for (n, e) in network_sizes]
y_values: list[np.ndarray] = []
y_mean: list[float] = []
y_std: list[float] = []

for nnodes, nedges in network_sizes:
    values = np.zeros(args.runs, dtype=np.float64)
    for j in range(args.runs):
        seed = SEED_BASE + j
        filename = f"{args.qchannel_capacity}-{nnodes}-{nedges}-{seed}.json"
        with open(os.path.join(args.indir, filename), "r") as file:
            data = json.load(file)
            values[j] = data["time_spent"]

    y_values.append(values)
    y_mean.append(np.mean(values).item())
    y_std.append(np.std(values).item())

# Save CSV output, if requested.
if args.csv:
    df = pd.DataFrame(
        {
            "network_size": network_sizes,
            "time_mean": y_mean,
            "time_std": y_std,
            "time_values": y_values,
        }
    )
    df.to_csv(args.csv, index=False)

# Generate simulation execution time plot.
mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 16,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "errorbar.capsize": 4,
    }
)

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.errorbar(x_ticks, y_mean, yerr=y_std, marker=None, linestyle="-", label="Time")
# ax.set_title("Average Execution Time")
ax.set_ylabel("Time (s)")
ax.set_xlabel("Network size (#nodes,#edges)")
ax.set_xticks(x_ticks, x_ticklabels, rotation=45, ha="right")
ax.grid(True, alpha=0.4)

plt_save(args.plt)
