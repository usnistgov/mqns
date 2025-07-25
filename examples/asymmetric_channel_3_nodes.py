import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from qns.entity.qchannel import LinkType
from qns.network.network import QuantumNetwork
from qns.network.protocol import ProactiveForwarder
from qns.simulator import Simulator
from qns.utils import log, set_seed

from examples_common.stats import gather_etg_decoh
from examples_common.topo_asymmetric_channel import build_topology


# Command line arguments
class Args(Tap):
    runs: int = 3  # number of trials per parameter set
    json: str = ""  # save results as JSON file
    plt: str = ""  # save plot as image file


args = Args().parse_args()

log.set_default_level("CRITICAL")

SEED_BASE = 100

# parameters
sim_duration = 3


def run_simulation(
    nodes: list[str],
    mem_capacities: list[int],
    ch_lengths: list[float],
    ch_capacities: list[tuple[int, int]],
    link_architectures: list[LinkType],
    t_coherence: float,
    seed: int,
):
    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = build_topology(
        nodes=nodes,
        mem_capacities=mem_capacities,
        ch_lengths=ch_lengths,
        ch_capacities=ch_capacities,
        t_coherence=t_coherence,
        link_architectures=link_architectures,
        swapping_order="swap_1",
    )
    net = QuantumNetwork(topo=topo)
    net.install(s)

    s.run()

    #### get stats
    _, _, decoh_ratio = gather_etg_decoh(net)
    fw_s = net.get_node("S").get_app(ProactiveForwarder)
    e2e_rate = fw_s.cnt.n_consumed / sim_duration
    mean_fidelity = fw_s.cnt.consumed_avg_fidelity
    return e2e_rate, decoh_ratio, mean_fidelity


########################### Main #########################

# Constants
SEED_BASE = 42
TOTAL_QUBITS = 6

ch_lengths: list[float] = [20, 20]

# Experiment parameters
t_cohere_values = [1e-3, 10e-3]
mem_allocs = [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]
mem_labels = [str(m) for m in mem_allocs]

channel_configs = {
    #  "DIM-DIM": [LinkType.DIM_BK, LinkType.DIM_BK],  # [25, 25]
    "SR-SR": [LinkType.SR, LinkType.SR],  # [25, 25]
    "SIM-DIM": [LinkType.SIM, LinkType.DIM_BK],  # [32, 18]
    "DIM-SIM": [LinkType.DIM_BK, LinkType.SIM],  # [18, 32]
}

# Store results: results[t_cohere][length_label] = dict of lists
results = {
    t: {length_label: {"rate_mean": [], "rate_std": [], "fid_mean": [], "fid_std": []} for length_label in channel_configs}
    for t in t_cohere_values
}

for t_cohere in t_cohere_values:
    for length_label, link_architectures in channel_configs.items():
        for left, right in mem_allocs:
            rates = []
            fids = []
            for i in range(args.runs):
                print(f"{length_label}, T_cohere={t_cohere:.3f}, Mem alloc={[left, right]}, run {i + 1}")
                seed = SEED_BASE + i

                ch_capacities = [(TOTAL_QUBITS, left), (right, TOTAL_QUBITS)]

                rate, *_, fidelity = run_simulation(
                    nodes=["S", "R", "D"],
                    mem_capacities=[TOTAL_QUBITS, TOTAL_QUBITS, TOTAL_QUBITS],
                    ch_lengths=ch_lengths,
                    ch_capacities=ch_capacities,
                    link_architectures=link_architectures,
                    t_coherence=t_cohere,
                    seed=seed,
                )
                rates.append(rate)
                fids.append(fidelity)

            res = results[t_cohere][length_label]
            res["rate_mean"].append(np.mean(rates))
            res["rate_std"].append(np.std(rates))
            res["fid_mean"].append(np.mean(fids))
            res["fid_std"].append(np.std(fids))

if args.json:
    with open(args.json, "w") as file:
        json.dump(results, file)

########################### Plot: Entanglement Rate #########################


# Update font and figure styling for academic readability at small dimensions
mpl.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 22,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "errorbar.capsize": 4,
    }
)

# Redraw plot with updated font sizes
fig_combined, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

for idx, t_cohere in enumerate(t_cohere_values):
    row_rate = 0
    row_fid = 1
    col = idx

    # Entanglement Rate
    ax_rate = axs[row_rate][col]
    for length_label in channel_configs:
        res = results[t_cohere][length_label]
        ax_rate.errorbar(mem_labels, res["rate_mean"], yerr=res["rate_std"], fmt="o--", capsize=4, label=length_label)
    ax_rate.set_title(f"T_cohere: {int(t_cohere * 1e3)} ms")
    ax_rate.set_ylabel("Ent. per second")
    ax_rate.grid(True, which="both", ls="--", lw=0.6, alpha=0.8)
    if col == 1:
        ax_rate.legend(loc="lower right")

    # Fidelity
    ax_fid = axs[row_fid][col]
    for length_label in channel_configs:
        res = results[t_cohere][length_label]
        ax_fid.errorbar(mem_labels, res["fid_mean"], yerr=res["fid_std"], fmt="s--", capsize=4, label=length_label)
    ax_fid.set_xlabel("Memory Allocation")
    ax_fid.set_ylabel("Fidelity")
    ax_fid.grid(True, which="both", ls="--", lw=0.6, alpha=0.8)

# fig_combined.suptitle("Entanglement Rate and Fidelity vs Memory Allocation", fontsize=22)
fig_combined.tight_layout(rect=(0, 0, 1, 0.95))
if args.plt:
    plt.savefig(args.plt, dpi=300, transparent=True)
plt.show()
