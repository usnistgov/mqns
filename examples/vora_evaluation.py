"""
This script compares entanglement swapping orders on a linear path.
The scenario is described in "Towards Optimal Orders for Entanglement Swapping in Path Graphs: A Greedy Approach"
https://doi.org/10.48550/arXiv.2504.14040 , Section V-D.

The path has an end-to-end distance of 150 km.
There are 3, 4, or 5 repeaters along the path, with the distances among them allocated with different weights:

* Uniform: all qchannels have the same length.
* Increasing: first qchannel has length `d0`; link i has length `(2i+1)d0`.
* Decreasing: reverse of Increasing.
* Mid-bottleneck: middle qchannel(s) are 1.2 times longer than all other qchannels.

There are 25 memory pairs for each link, and the memory coherence time is 10 ms.

Entanglement swapping orders under comparison are:

* asap: swap as soon as possible.
* l2r: swap left-to-right.
* baln: swap in balanced tree.
* vora: swap order generated with vora algorithm based on training data.

``vora_evaluation.voraswap.json`` contains precomputed vora swapping orders for the paths used in this script.
It is only compatible with the total distance, number of memory pairs, and memory coherence time mentioned above.
To alter these parameters, you must use ``--vora_train`` flag to re-generate the vora swapping orders,
and then use ``--vora_load`` flag to load the new data file containing vora swapping orders::

    # Start the training with --vora_train flag.
    python vora_evaluation.py --vora_train \\
        --runs 1000 --total_distance 150 --t_cohere 0.010 --qchannel_capacity 25
    # This command prints a script that calls linear_attempts.py to generate training data,
    # which is then fed back to vora_evaluation.py --vora_regen to compute the vora swapping orders.
    # Review the script, set OUTDIR variable, and then run the script.

    # Now you can load the new data file with --vora_load flag and matching parameters.
    python vora_evaluation.py --vora_load voraswap.json \\
        --total_distance 150 --t_cohere 0.010 --qchannel_capacity 25 \\
        --csv result.csv --plt result.png
"""

import copy
import itertools
import json
import os.path
from collections.abc import Callable
from multiprocessing import Pool, freeze_support
from typing import cast

import numpy as np
import pandas as pd
from tap import Tap

from mqns.network.network import QuantumNetwork
from mqns.network.proactive import ProactiveForwarder, compute_vora_swap_sequence
from mqns.network.topology.topo import Topology
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from examples_common.plotting import Axes2D, plt, plt_save
from examples_common.stats import gather_etg_decoh
from examples_common.topo_linear import build_topology

log.set_default_level("CRITICAL")


NUM_ROUTERS_OPTIONS = [3, 4, 5]
DIST_PROPORTIONS = ["decreasing", "increasing", "mid_bottleneck", "uniform"]
SWAP_CONFIGS = ["asap", "baln", "vora", "l2r"]


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
"""


class ParameterSet:
    def __init__(self):
        self.seed_base = 100
        self.n_runs = 10

        self.sim_duration = 5.0
        self.total_distance = 150  # km
        self.t_cohere = 0.01  # sec
        self.qchannel_capacity = 25

        self._precomputed_vora: dict = {}
        self.number_of_routers: int = 0
        self.distance_proportion: str = "invalid"
        self.swapping_config: str = "invalid"

    def clone_with(self, number_of_routers: int, distance_proportion: str, swapping_config: str) -> "ParameterSet":
        p = copy.copy(self)
        p.number_of_routers = number_of_routers
        p.distance_proportion = distance_proportion
        p.swapping_config = swapping_config
        return p

    @property
    def t_cohere_ns(self) -> int:
        return int(self.t_cohere * 1e9)

    def build_topology(self) -> Topology:
        distances = self.compute_distances()
        swap = self.get_swap_sequence()
        log.info(f"build_topology: distances={distances} sum={sum(distances)} swap-sequence={swap}")

        return build_topology(
            nodes=2 + self.number_of_routers,
            t_cohere=self.t_cohere,
            channel_length=distances,
            channel_capacity=self.qchannel_capacity,
            swap=swap,
        )

    def compute_distances(self) -> list[float]:
        n_segments = self.number_of_routers + 1
        weights = DISTANCE_PROPORTION_WEIGHTS[self.distance_proportion](n_segments)
        sum_weight = sum(weights)
        return [self.total_distance * w / sum_weight for w in weights]

    def to_linear_attempts_csv_filename(self, train_qchannel_capacity=1) -> str:
        return f"{self.number_of_routers}-{self.distance_proportion}-{train_qchannel_capacity}.csv"

    def load_vora(self, filename: str) -> None:
        """
        Load precomputed vora swapping orders and validate that they match the topology parameters.
        """
        with open(filename) as f:
            d = json.load(f)
        assert type(d) is dict, "invalid precomputed vora swapping orders"
        for k in "total_distance", "t_cohere_ns", "qchannel_capacity":
            assert d[k] == getattr(self, k), f"mismatched {k} in precomputed vora swapping orders"
        for n, p in itertools.product(NUM_ROUTERS_OPTIONS, DIST_PROPORTIONS):
            assert type(d.get(f"{n}-{p}", None)) is list, f"missing {n}-{p} in precomputed vora swapping orders"
        self._precomputed_vora = d

    def get_swap_sequence(self) -> str | list[int]:
        if self.swapping_config != "vora":
            return f"swap_{self.number_of_routers}_{self.swapping_config}"
        return self._precomputed_vora[f"{self.number_of_routers}-{self.distance_proportion}"]


def vora_train_row(p: ParameterSet, num_routers: int, dist_prop: str) -> str:
    """
    Generate linear_attempts.py command line for collecting training data for the given topology.
    """
    p = p.clone_with(num_routers, dist_prop, "no_swap")

    distances = p.compute_distances()
    L = " ".join([str(d) for d in distances])
    filename = p.to_linear_attempts_csv_filename()
    return f"python linear_attempts.py --runs {p.n_runs} --L {L} --M 1 --csv $OUTDIR/{filename}"


def vora_regen_row(p: ParameterSet, num_routers: int, dist_prop: str, indir: str) -> list[int]:
    """
    Regenerate vora swapping order from the output of linear_attempts.py.
    """
    p = p.clone_with(num_routers, dist_prop, "no_swap")
    data = pd.read_csv(os.path.join(indir, p.to_linear_attempts_csv_filename()))
    return compute_vora_swap_sequence(
        lengths=list(data["L"]),
        attempts=list(data["Attempts rate"]),
        success=list(data["Success rate"]),
        t_cohere=p.t_cohere,
        qchannel_capacity=p.qchannel_capacity,
    )


def run_simulation(p: ParameterSet, seed: int) -> tuple[float, float]:
    set_seed(seed)

    topo = p.build_topology()
    net = QuantumNetwork(topo)

    s = Simulator(0, p.sim_duration + 5e-06, accuracy=1000000, install_to=(log, net))
    s.run()

    #### get stats
    _, total_decohered, _ = gather_etg_decoh(net)
    e2e_count = net.get_node("S").get_app(ProactiveForwarder).cnt.n_consumed
    return e2e_count / p.sim_duration, total_decohered / e2e_count if e2e_count > 0 else 0


def run_row(p: ParameterSet, num_routers: int, dist_prop: str, swap_conf: str) -> dict:
    """
    Run simulations for one parameter set.
    """
    p = p.clone_with(num_routers, dist_prop, swap_conf)

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
    axes = cast(Axes2D, axes)

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
    plt_save(save_plt)


if __name__ == "__main__":
    freeze_support()
    p = ParameterSet()

    class Args(Tap):
        vora_train: bool = False  # generate training script for vora swapping sequences
        vora_regen: str = ""  # regenerate vora swapping sequences from training data directory
        vora_load: str = ""  # load vora swapping sequences from file
        workers: int = 1  # number of workers for parallel execution
        runs: int = p.n_runs  # number of trials per parameter set
        sim_duration: float = p.sim_duration  # simulation duration in seconds
        total_distance: int = p.total_distance  # total distance
        t_cohere: float = p.t_cohere  # memory coherence time
        qchannel_capacity: int = p.qchannel_capacity  # qchannel capacity
        csv: str = ""  # save results as CSV file
        plt: str = ""  # save plot as image file

    args = Args().parse_args()
    p.n_runs = args.runs
    p.sim_duration = args.sim_duration
    p.total_distance = args.total_distance
    p.t_cohere = args.t_cohere
    p.qchannel_capacity = args.qchannel_capacity

    if args.vora_train:
        # Generate training script.
        script = itertools.chain(
            (vora_train_row(*a) for a in itertools.product([p], NUM_ROUTERS_OPTIONS, DIST_PROPORTIONS)),
            [
                f"python vora_evaluation.py --vora_regen $OUTDIR"
                f" --total_distance {p.total_distance}"
                f" --t_cohere {p.t_cohere}"
                f" --qchannel_capacity {p.qchannel_capacity}"
                f" >$OUTDIR/voraswap.json\n"
            ],
        )
        print("\n".join(script))

    elif args.vora_regen:
        # Generate vora swapping orders based on training data.
        d: dict = {
            f"{n}-{d}": vora_regen_row(p, n, d, indir=args.vora_regen)
            for n, d in itertools.product(NUM_ROUTERS_OPTIONS, DIST_PROPORTIONS)
        }
        d["total_distance"] = p.total_distance
        d["t_cohere_ns"] = p.t_cohere_ns
        d["qchannel_capacity"] = p.qchannel_capacity
        print(json.dumps(d))

    else:
        # Load precomputed vora swapping orders.
        p.load_vora(args.vora_load or os.path.join(os.path.dirname(__file__), "vora_evaluation.voraswap.json"))

        # Run simulations.
        with Pool(processes=args.workers) as pool:
            results = pool.starmap(run_row, itertools.product([p], NUM_ROUTERS_OPTIONS, DIST_PROPORTIONS, SWAP_CONFIGS))

        save_results(results, save_csv=args.csv, save_plt=args.plt)
