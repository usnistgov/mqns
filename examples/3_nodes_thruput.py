import itertools
from multiprocessing import Pool, freeze_support
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

from mqns.network.network import QuantumNetwork
from mqns.network.proactive import ProactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from examples_common.topo_3_nodes import build_topology

log.set_default_level("CRITICAL")

# Constants
SEED_BASE = 100
sim_duration = 3

# Experiment parameters
t_cohere_values = [0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1]


def run_simulation(t_coherence: float, seed: int):
    """Run a simulation with a given coherence time and seed.

    This function sets up and executes a simulation using:
      - A generated topology based on the specified qubit coherence time,
      - A quantum network with Dijkstra-based routing algorithm, and asynchronous timing mode,
      - A seeded random number generator.

    After simulation, it gathers statistics including:
      - Total number of successful entanglement generations,
      - Total number of decohered qubits,
      - End-to-end entanglement rate between source node "S" and destination node "D".

    Args:
        t_coherence (float): Qubit coherence time (in seconds), used to define memory decoherence rate.
        seed (int): Seed for the random number generator.

    Returns:
        Tuple[float, float]:
            - `e2e_rate`: End-to-end entanglement generation rate (entangled pairs per second).
            - `decoherence_ratio`: Fraction of entangled qubits that decohered before use
            over the number of e2e entanglements generated.

    """
    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = build_topology(t_coherence=t_coherence, channel_qubits=1)
    net = QuantumNetwork(topo=topo)
    net.install(s)

    s.run()

    #### get stats
    e2e_rate = net.get_node("S").get_app(ProactiveForwarder).cnt.n_consumed / sim_duration
    return e2e_rate


def run_row(n_runs: int, t_cohere: float) -> tuple[float, list[float]]:
    rates: list[float] = []
    for i in range(n_runs):
        print(f"T_cohere={t_cohere:.4f}, run {i + 1}")
        rate = run_simulation(t_cohere, SEED_BASE + i)
        rates.append(rate)
    return t_cohere, rates


def save_results(results: Any, *, save_csv: str | None, save_plt: str | None):
    # Final results summary print
    print("\nT_coh    Rate")
    for t, mean, std in zip(results["T_cohere"], results["Mean Rate"], results["Std Rate"]):
        print(f"{t:<7.3f}  {mean:>5.1f} ({std:.1f})")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    if save_csv:
        df.to_csv(save_csv, index=False)

    # Plotting
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
    if save_plt:
        plt.savefig(save_plt, dpi=300, transparent=True)
    plt.show()


if __name__ == "__main__":
    freeze_support()

    # Command line arguments
    class Args(Tap):
        workers: int = 1  # number of workers for parallel execution
        runs: int = 100  # number of trials per parameter set
        csv: str = ""  # save results as CSV file
        plt: str = ""  # save plot as image file

    args = Args().parse_args()

    # Simulator loop with process-based parallelism
    with Pool(processes=args.workers) as pool:
        rows = pool.starmap(run_row, itertools.product([args.runs], t_cohere_values))

    results = {"T_cohere": [], "Mean Rate": [], "Std Rate": []}
    for t_cohere, rates in rows:
        results["T_cohere"].append(t_cohere)
        results["Mean Rate"].append(np.mean(rates))
        results["Std Rate"].append(np.std(rates))

    save_results(results, save_csv=args.csv, save_plt=args.plt)
