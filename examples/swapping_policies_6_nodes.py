"""
This script evaluates entanglement swapping strategies combined with memory allocation in a linear six-node network:
``S-R1-R2-R3-R4-D``, with link lengths of 32 km, 18 km, 35 km, 16 km, and 24 km.
Each node has six quantum memories.

It compares two memory allocation schemes: a uniform configuration ``[3, 3, 3, 3, 3]``
and a non-uniform one ``[4, 2, 4, 2, 4]``, representing the number of qubits assigned to each channel.
Coherence times are set to 5 ms, 10 ms, or 20 ms.

Five swapping strategies are tested:

• asap: swap immediately as EPRs become available
• baln: swap R1 and R3 first, then R2, then R4
• baln2: swap R2 and R4 first, then R3, then R1
• l2r: swap sequentially from R1 to R4
• r2l: swap sequentially from R4 to R1

The results are end-to-end throughput of each scenario.
"""

import json
from typing import cast

import numpy as np
from tap import Tap

from mqns.network.network import QuantumNetwork
from mqns.network.proactive import ProactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from examples_common.plotting import Axes1D, mpl, plt, plt_save
from examples_common.stats import gather_etg_decoh
from examples_common.topo_linear import build_topology

log.set_default_level("CRITICAL")


SEED_BASE = 100
N_NODES = 6
TOTAL_QUBITS = 6
CHANNEL_LENGTHS: list[float] = [32, 18, 35, 16, 24]

# parameters
sim_duration = 3


def run_simulation(
    ch_capacities: list[tuple[int, int]],
    t_cohere: float,
    swapping_order: str,
    seed: int,
):
    set_seed(seed)

    topo = build_topology(
        nodes=N_NODES,
        mem_capacity=TOTAL_QUBITS,
        t_cohere=t_cohere,
        channel_length=CHANNEL_LENGTHS,
        channel_capacity=ch_capacities,
        swap=swapping_order,
    )
    net = QuantumNetwork(topo)

    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000, install_to=(log, net))
    s.run()

    #### get stats
    _, _, decoh_ratio = gather_etg_decoh(net)
    fw_s = net.get_node("S").get_app(ProactiveForwarder)
    e2e_rate = fw_s.cnt.n_consumed / sim_duration
    mean_fidelity = fw_s.cnt.consumed_avg_fidelity
    return e2e_rate, decoh_ratio, mean_fidelity


Results = dict[str, dict[str, dict[float, tuple[float, float]]]]


def plot_results(results: Results) -> None:
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
    axs = cast(Axes1D, axs)

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
    plt_save(args.plt)


########################### Main #########################


if __name__ == "__main__":

    class Args(Tap):
        runs: int = 3  # number of trials per parameter set
        json: str = ""  # save results as JSON file
        plt: str = ""  # save plot as image file

    args = Args().parse_args()

    # Define swapping policies to test
    swapping_order_configs = ["swap_4_baln", "swap_4_baln2", "swap_4_l2r", "swap_4_r2l", "swap_4_asap"]
    ch_capacities_configs = {
        "[3, 3, 3, 3, 3]": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
        "[4, 2, 4, 2, 4]": [(4, 4), (2, 2), (4, 4), (2, 2), (4, 4)],
    }
    t_cohere_values = [5e-3, 10e-3, 20e-3]

    # Store results: mem_label -> policy -> t_cohere -> list of rates
    results: Results = {
        mem_label: {order: {t: (0.0, 0.0) for t in t_cohere_values} for order in swapping_order_configs}
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
                        ch_capacities=ch_capacities,
                        t_cohere=t_cohere,
                        swapping_order=swapping_config,
                        seed=seed,
                    )
                    run_rates.append(rate)

                results[mem_label][order][t_cohere] = (np.mean(run_rates).item(), np.std(run_rates).item())

    if args.json:
        with open(args.json, "w") as file:
            json.dump(results, file)

    plot_results(results)
