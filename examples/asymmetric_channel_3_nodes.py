import json

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
    plt_rate: str = ""  # save entanglement rate plot as image file
    plt_fid: str = ""  # save fidelity plot as image file


args = Args().parse_args()

log.set_default_level("CRITICAL")

SEED_BASE = 100

# parameters
sim_duration = 3


def run_simulation(
    nodes: list[str],
    mem_capacities: list[int],
    channel_lengths: list[float],
    capacities: list[tuple[int, int]],
    t_coherence: float,
    seed: int,
):
    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = build_topology(
        nodes=nodes,
        mem_capacities=mem_capacities,
        channel_lengths=channel_lengths,
        capacities=capacities,
        t_coherence=t_coherence,
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

# Experiment parameters
t_cohere_values = [5e-3, 10e-3, 20e-3]
mem_allocs = [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]
mem_labels = [str(m) for m in mem_allocs]

channel_configs: dict[str, list[float]] = {
    "Equal": [25, 25],
    "L1 > L2": [32, 18],
    "L1 < L2": [18, 20],
}

# Store results: results[t_cohere][length_label] = dict of lists
results = {
    t: {length_label: {"rate_mean": [], "rate_std": [], "fid_mean": [], "fid_std": []} for length_label in channel_configs}
    for t in t_cohere_values
}

for t_cohere in t_cohere_values:
    for length_label, ch_lengths in channel_configs.items():
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
                    channel_lengths=ch_lengths,
                    capacities=ch_capacities,
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

fig_rate, axs_rate = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for idx, t_cohere in enumerate(t_cohere_values):
    ax = axs_rate[idx]
    for length_label in channel_configs:
        res = results[t_cohere][length_label]
        ax.errorbar(mem_labels, res["rate_mean"], yerr=res["rate_std"], fmt="o--", capsize=4, label=length_label)
    ax.set_title(f"T_cohere: {int(t_cohere * 1e3)} ms")
    ax.set_xlabel("Mem alloc.")
    if idx == 0:
        ax.set_ylabel("Ent. per second")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

fig_rate.suptitle("End-to-end Entanglement Rate vs Memory Allocation", fontsize=14)
fig_rate.tight_layout()
if args.plt_rate:
    fig_rate.savefig(args.plt_rate, dpi=300, transparent=True)
plt.show()

########################### Plot: Fidelity #########################

fig_fid, axs_fid = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for idx, t_cohere in enumerate(t_cohere_values):
    ax = axs_fid[idx]
    for length_label in channel_configs:
        res = results[t_cohere][length_label]
        ax.errorbar(mem_labels, res["fid_mean"], yerr=res["fid_std"], fmt="s--", capsize=4, label=length_label)
    ax.set_title(f"T_cohere: {int(t_cohere * 1e3)} ms")
    ax.set_xlabel("Mem alloc.")
    if idx == 0:
        ax.set_ylabel("Fidelity")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

fig_fid.suptitle("End-to-end Fidelity vs Memory Allocation", fontsize=14)
fig_fid.tight_layout()
if args.plt_fid:
    fig_fid.savefig(args.plt_fid, dpi=300, transparent=True)
plt.show()
