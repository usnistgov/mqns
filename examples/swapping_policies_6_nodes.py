import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

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
    t_coherence: float,
    swapping_order: str,
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
        swapping_order=swapping_order,
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

# Define swapping policies to test
swapping_order_configs = ["swap_4_baln", "swap_4_baln2", "swap_4_l2r", "swap_4_r2l", "swap_4_asap"]
ch_capacities_configs = {
    "[3, 3, 3, 3, 3]": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
    "[4, 2, 4, 2, 4]": [(4, 4), (2, 2), (4, 4), (2, 2), (4, 4)],
}
t_cohere_values = [5e-3, 10e-3, 20e-3]

nodes = ["S", "R1", "R2", "R3", "R4", "D"]
mem_capacities = [6, 6, 6, 6, 6, 6]
channel_lengths: list[float] = [32, 18, 35, 16, 24]
TOTAL_QUBITS = 6

SEED_BASE = 100

# Store results: mem_label -> policy -> t_cohere -> list of rates
results = {
    mem_label: {order: {t: [] for t in t_cohere_values} for order in swapping_order_configs}
    for mem_label in ch_capacities_configs
}

for mem_label, mem_allocs in ch_capacities_configs.items():
    for order in swapping_order_configs:
        print(f"\n>>> Testing order: {order}, Channel Mem allocation: {mem_label}")
        for t_cohere in t_cohere_values:
            run_rates = []
            for i in range(args.runs):
                seed = SEED_BASE + i
                swapping_config = order

                ch_capacities = mem_allocs

                rate, *_ = run_simulation(
                    nodes=nodes,
                    mem_capacities=mem_capacities,
                    ch_lengths=channel_lengths,
                    ch_capacities=ch_capacities,
                    t_coherence=t_cohere,
                    swapping_order=swapping_config,
                    seed=seed,
                )
                run_rates.append(rate)

            results[mem_label][order][t_cohere] = (np.mean(run_rates), np.std(run_rates))

if args.json:
    with open(args.json, "w") as file:
        json.dump(results, file)


# Reapply font and style settings for academic clarity
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
        "lines.markersize": 7,
        "errorbar.capsize": 4,
    }
)

# Create plot with updated style
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

colors = {"baln": "green", "baln2": "red", "l2r": "purple", "r2l": "brown", "asap": "pink"}
markers = {"baln": "v", "baln2": "s", "l2r": "^", "r2l": "x", "asap": "d"}

for ax_idx, (mem_label, policy_dict) in enumerate(results.items()):
    ax = axs[ax_idx]
    for full_policy, t_dict in policy_dict.items():
        policy = full_policy.replace("swap_4_", "")
        means = [t_dict[t][0] for t in t_cohere_values]
        stds = [t_dict[t][1] for t in t_cohere_values]
        ax.errorbar(
            [t * 1e3 for t in t_cohere_values],
            means,
            yerr=stds,
            fmt=markers[policy],
            linestyle="-",
            capsize=4,
            label=policy,
            color=colors[policy],
        )
    ax.set_title(f"Mem. Alloc: {mem_label}")
    ax.set_xlabel("T_cohere (ms)")
    if ax_idx == 0:
        ax.set_ylabel("Throughput (eps)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.8)

axs[-1].legend(title="Policy", loc="lower right")
plt.tight_layout()
if args.plt:
    plt.savefig(args.plt, dpi=300, transparent=True)
plt.show()
