import itertools
import json
from multiprocessing import Pool, freeze_support
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from mqns.entity.qchannel import LinkArch, LinkArchDimBk, LinkArchSim, LinkArchSr
from mqns.network.network import QuantumNetwork
from mqns.network.proactive import ProactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from examples_common.topo_asymmetric_channel import build_topology

log.set_default_level("CRITICAL")

# Constants
sim_duration = 3
SEED_BASE = 42
ch_lengths: list[float] = [20, 20]

# Experiment parameters
t_cohere_values = [1e-3, 10e-3]
mem_allocs = [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]
mem_labels = [str(m) for m in mem_allocs]
channel_configs: dict[str, list[LinkArch]] = {
    "SR-SR": [LinkArchSr(), LinkArchSr()],
    "SIM-DIM": [LinkArchSim(), LinkArchDimBk()],
    "DIM-SIM": [LinkArchDimBk(), LinkArchSim()],
}


def run_simulation(
    nodes: list[str],
    mem_capacities: list[int],
    ch_lengths: list[float],
    ch_capacities: list[tuple[int, int]],
    link_architectures: list[LinkArch],
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
    fw_s = net.get_node("S").get_app(ProactiveForwarder)
    e2e_rate = fw_s.cnt.n_consumed / sim_duration
    mean_fidelity = fw_s.cnt.consumed_avg_fidelity
    return e2e_rate, mean_fidelity


def run_row(
    n_runs: int, t_cohere: float, mem_alloc: tuple[int, int], arch: tuple[str, list[LinkArch]]
) -> tuple[float, str, list[float], list[float]]:
    left, right = mem_alloc
    total_qubits = left + right
    arch_label, link_archs = arch

    rates: list[float] = []
    fids: list[float] = []
    for i in range(n_runs):
        print(f"{arch_label}, T_cohere={t_cohere:.3f}, Mem alloc={[left, right]}, run {i + 1}")
        rate, fidelity = run_simulation(
            nodes=["S", "R", "D"],
            mem_capacities=[total_qubits, total_qubits, total_qubits],
            ch_lengths=ch_lengths,
            ch_capacities=[(total_qubits, left), (right, total_qubits)],
            link_architectures=link_archs,
            t_coherence=t_cohere,
            seed=SEED_BASE + i,
        )
        rates.append(rate)
        fids.append(fidelity)
    return t_cohere, arch_label, rates, fids


def save_results(results: Any, *, save_json: str | None, save_plt: str | None):
    if save_json:
        with open(save_json, "w") as file:
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
        for arch_label in channel_configs:
            res = results[t_cohere][arch_label]
            ax_rate.errorbar(mem_labels, res["rate_mean"], yerr=res["rate_std"], fmt="o--", capsize=4, label=arch_label)
        ax_rate.set_title(f"T_cohere: {int(t_cohere * 1e3)} ms")
        ax_rate.set_ylabel("Ent. per second")
        ax_rate.grid(True, which="both", ls="--", lw=0.6, alpha=0.8)
        if col == 1:
            ax_rate.legend(loc="lower right")

        # Fidelity
        ax_fid = axs[row_fid][col]
        for arch_label in channel_configs:
            res = results[t_cohere][arch_label]
            ax_fid.errorbar(mem_labels, res["fid_mean"], yerr=res["fid_std"], fmt="s--", capsize=4, label=arch_label)
        ax_fid.set_xlabel("Memory Allocation")
        ax_fid.set_ylabel("Fidelity")
        ax_fid.grid(True, which="both", ls="--", lw=0.6, alpha=0.8)

    # fig_combined.suptitle("Entanglement Rate and Fidelity vs Memory Allocation", fontsize=22)
    fig_combined.tight_layout(rect=(0, 0, 1, 0.95))
    if save_plt:
        plt.savefig(save_plt, dpi=300, transparent=True)
    plt.show()


if __name__ == "__main__":
    freeze_support()

    # Command line arguments
    class Args(Tap):
        workers: int = 1  # number of workers for parallel execution
        runs: int = 3  # number of trials per parameter set
        json: str = ""  # save results as JSON file
        plt: str = ""  # save plot as image file

    args = Args().parse_args()

    # Simulator loop with process-based parallelism
    with Pool(processes=args.workers) as pool:
        rows = pool.starmap(run_row, itertools.product([args.runs], t_cohere_values, mem_allocs, channel_configs.items()))

    # Store results: results[t_cohere][arch_label] = dict of lists
    results = {
        t: {arch_label: {"rate_mean": [], "rate_std": [], "fid_mean": [], "fid_std": []} for arch_label in channel_configs}
        for t in t_cohere_values
    }
    for t_cohere, arch_label, rates, fids in rows:
        res = results[t_cohere][arch_label]
        res["rate_mean"].append(np.mean(rates))
        res["rate_std"].append(np.std(rates))
        res["fid_mean"].append(np.mean(fids))
        res["fid_std"].append(np.std(fids))

    save_results(results, save_json=args.json, save_plt=args.plt)
