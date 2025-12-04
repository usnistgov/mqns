import itertools
import sys
from collections.abc import Callable
from copy import deepcopy
from multiprocessing import Pool, freeze_support

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

from mqns.network.network import QuantumNetwork
from mqns.network.proactive import ProactiveForwarder
from mqns.network.topology.topo import Topology
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from examples_common.stats import gather_etg_decoh
from examples_common.topo_linear import build_topology

log.set_default_level("CRITICAL")


def distance_proportion_weights_mid_bottleneck(n: int) -> list[float]:
    n_mid = 2 if n % 2 == 0 else 1
    n_side = (n - n_mid) // 2
    return [1.0] * n_side + [1.2] * n_mid + [1.0] * n_side


DISTANCE_PROPORTION_WEIGHTS: dict[str, Callable[[int], list[float]]] = {
    "uniform": lambda n: [1.0] * n,
    "increasing": lambda n: [i * 2 + 1.0 for i in range(n)],
    "decreasing": lambda n: [i * 2 + 1.0 for i in range(n)][::-1],
    "mid_bottleneck": distance_proportion_weights_mid_bottleneck,
}
"""
For each distance proportion, function to generate relative weights of qchannel lengths, given number of qchannel segments.

Uniform: all qchannels have the same length.
Increasing: first qchannel has length `d0`; link i has length `(2i+1)d0`.
Decreasing: reverse of Increasing.
Mid-bottleneck: middle qchannel(s) are 1.2 times longer than all other qchannels.
"""

VORA_SWAPPING_ORDER: dict[int, dict[str, list[int]]] = {
    3: {
        "uniform": [1, 3, 2],
        "increasing": [1, 2, 3],
        "decreasing": [3, 2, 1],
        "mid_bottleneck": [1, 3, 2],
    },
    4: {
        "uniform": [1, 4, 2, 3],
        "increasing": [1, 2, 3, 4],
        "decreasing": [4, 3, 2, 1],
        "mid_bottleneck": [1, 3, 4, 2],
    },
    5: {
        "uniform": [1, 4, 2, 5, 3],
        "increasing": [1, 2, 3, 4, 5],
        "decreasing": [5, 4, 3, 2, 1],
        "mid_bottleneck": [1, 4, 3, 5, 2],
    },
}
"""
Pre-computed VoraSwap swapping orders.
"""


class ParameterSet:
    def __init__(self):
        self.seed_base = 100
        self.sim_duration = 5.0
        self.channel_qubits = 25
        self.t_coherence = 0.01  # sec

        self.total_distance = 150  # km

        self.n_runs = 10

        self.number_of_routers: int
        self.distance_proportion: str
        self.swapping_config: str

    def build_topology(self) -> Topology:
        distances = self.compute_distances()
        swap = self.get_swap_sequence()
        log.info(f"build_topology: distances={distances} sum={sum(distances)} swap-sequence={swap}")

        return build_topology(
            nodes=2 + self.number_of_routers,
            t_coherence=self.t_coherence,
            channel_length=distances,
            channel_capacity=self.channel_qubits,
            swap=swap,
        )

    def compute_distances(self) -> list[float]:
        n_segments = self.number_of_routers + 1
        weights = DISTANCE_PROPORTION_WEIGHTS[self.distance_proportion](n_segments)
        sum_weight = sum(weights)
        return [self.total_distance * w / sum_weight for w in weights]

    def get_swap_sequence(self) -> str | list[int]:
        if self.swapping_config != "vora":
            return f"swap_{self.number_of_routers}_{self.swapping_config}"

        so = VORA_SWAPPING_ORDER[self.number_of_routers][self.distance_proportion]
        sd_rank = max(so) + 1
        return [sd_rank] + so + [sd_rank]


def train_attempts_row(p: ParameterSet, num_routers: int, dist_prop: str) -> str:
    """
    Generate linear_attempts.py command line for collecting training data for the given topology.
    """
    p = deepcopy(p)
    p.number_of_routers = num_routers
    p.distance_proportion = dist_prop
    p.swapping_config = "no_swap"

    distances = p.compute_distances()
    L = " ".join([str(d) for d in distances])
    filename = f"{num_routers}-{dist_prop}-{p.channel_qubits}"
    return (
        f"python linear_attempts.py --runs {p.n_runs} --L {L} --M {p.channel_qubits} "
        f"--json $OUTDIR/{filename}.json --csv $OUTDIR/{filename}.csv\n"
    )


def run_simulation(p: ParameterSet, seed: int) -> tuple[float, float]:
    set_seed(seed)
    s = Simulator(0, p.sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = p.build_topology()
    net = QuantumNetwork(topo=topo)

    net.install(s)
    s.run()

    #### get stats
    _, total_decohered, _ = gather_etg_decoh(net)
    e2e_count = net.get_node("S").get_app(ProactiveForwarder).cnt.n_consumed
    return e2e_count / p.sim_duration, total_decohered / e2e_count if e2e_count > 0 else 0


def run_row(p: ParameterSet, num_routers: int, dist_prop: str, swap_conf: str) -> dict:
    """
    Run simulations for one parameter set.
    """
    p = deepcopy(p)
    p.number_of_routers = num_routers
    p.distance_proportion = dist_prop
    p.swapping_config = swap_conf

    entanglements: list[float] = []
    expired: list[float] = []
    for i in range(p.n_runs):
        print(f"Simulation: {num_routers} routers | {dist_prop} distances | {swap_conf} | run #{i + 1}")
        seed = p.seed_base + i
        e2e_count, expired_count = run_simulation(p, seed)
        entanglements.append(e2e_count)
        expired.append(expired_count)

    mean_entg = np.mean(entanglements).item()
    std_entg = np.std(entanglements).item()
    mean_exp = np.mean(expired).item()
    std_exp = np.std(expired).item()

    return {
        "Routers": num_routers,
        "Distance Distribution": dist_prop,
        "Swapping Config": swap_conf,
        "Entanglements Per Second": mean_entg,
        "Entanglements Std": std_entg,
        "Entanglements All Runs": entanglements,
        "Expired Memories Per Entanglement": mean_exp,
        "Expired Memories Std": std_exp,
        "Expired Memories All Runs": expired,
    }


def save_results(results: list[dict], *, save_csv: str, save_plt: str) -> None:
    df = pd.DataFrame(results)
    if save_csv:
        df.to_csv(save_csv, index=False)

    # === Combined Plot ===
    _, axes = plt.subplots(2, 3, figsize=(18, 10), sharey="row")

    x_labels = DIST_PROPORTIONS
    x = np.arange(len(x_labels))
    width = 0.2

    for i, num_routers in enumerate(NUM_ROUTERS_OPTIONS):
        df_subset = df[df["Routers"] == num_routers]

        # --- Top Row: Entanglements Per Second ---
        ax1 = axes[0, i]
        for j, swap_conf in enumerate(SWAP_CONFIGS):
            means = []
            stds = []
            for dist_prop in x_labels:
                row = df_subset[(df_subset["Distance Distribution"] == dist_prop) & (df_subset["Swapping Config"] == swap_conf)]
                means.append(row["Entanglements Per Second"].values[0])
                stds.append(row["Entanglements Std"].values[0])
            ax1.bar(x + j * width, means, width, yerr=stds, label=swap_conf)

        ax1.set_title(f"Entanglements/sec - {num_routers} Routers")
        ax1.set_xticks(x + 1.5 * width)
        ax1.set_xticklabels(x_labels)
        if i == 0:
            ax1.set_ylabel("Entanglements Per Second")

        # --- Bottom Row: Expired Memories Per Entanglement ---
        ax2 = axes[1, i]
        for j, swap_conf in enumerate(SWAP_CONFIGS):
            means = []
            stds = []
            for dist_prop in x_labels:
                row = df_subset[(df_subset["Distance Distribution"] == dist_prop) & (df_subset["Swapping Config"] == swap_conf)]
                means.append(row["Expired Memories Per Entanglement"].values[0])
                stds.append(row["Expired Memories Std"].values[0])
            ax2.bar(x + j * width, means, width, yerr=stds, label=swap_conf)

        ax2.set_title(f"Expired Memories/Entg - {num_routers} Routers")
        ax2.set_xticks(x + 1.5 * width)
        ax2.set_xticklabels(x_labels)
        if i == 0:
            ax2.set_ylabel("Expired Memories per Entanglement")

    # Add legends only once
    axes[0, 0].legend(loc="upper left")
    axes[1, 0].legend(loc="upper left")

    plt.tight_layout()
    if save_plt:
        plt.savefig(save_plt, dpi=300, transparent=True)
    plt.show()


NUM_ROUTERS_OPTIONS = [3, 4, 5]
DIST_PROPORTIONS = ["decreasing", "increasing", "mid_bottleneck", "uniform"]
SWAP_CONFIGS = ["asap", "baln", "vora", "l2r"]


if __name__ == "__main__":
    freeze_support()
    p = ParameterSet()

    # Command line arguments
    class Args(Tap):
        train_attempts: bool = False  # generate training script for linear_attempts.py
        workers: int = 1  # number of workers for parallel execution
        runs: int = p.n_runs  # number of trials per parameter set
        sim_duration: float = p.sim_duration  # simulation duration in seconds
        channel_qubits: int = p.channel_qubits  # qchannel capacity
        csv: str = ""  # save results as CSV file
        plt: str = ""  # save plot as image file

    args = Args().parse_args()
    p.n_runs = args.runs
    p.sim_duration = args.sim_duration
    p.channel_qubits = args.channel_qubits

    if args.train_attempts:
        script = (train_attempts_row(*a) for a in itertools.product([p], NUM_ROUTERS_OPTIONS, DIST_PROPORTIONS))
        sys.stdout.writelines(script)
        sys.exit()

    # Simulator loop with process-based parallelism
    with Pool(processes=args.workers) as pool:
        results = pool.starmap(run_row, itertools.product([p], NUM_ROUTERS_OPTIONS, DIST_PROPORTIONS, SWAP_CONFIGS))

    save_results(results, save_csv=args.csv, save_plt=args.plt)
