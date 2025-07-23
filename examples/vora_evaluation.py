import itertools
from copy import deepcopy
from multiprocessing import Pool, freeze_support

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

from qns.network import QuantumNetwork
from qns.network.protocol import LinkLayer, ProactiveForwarder, ProactiveRoutingControllerApp
from qns.network.topology.customtopo import CustomTopology, Topo, TopoCChannel, TopoController, TopoQChannel, TopoQNode
from qns.simulator import Simulator
from qns.utils import log, set_seed

from examples_common.stats import gather_etg_decoh

log.set_default_level("CRITICAL")


class ParameterSet:
    def __init__(self):
        self.seed_base = 100

        self.sim_duration = 5

        self.fiber_alpha = 0.2
        self.eta_d = 0.95
        self.eta_s = 0.95
        self.frequency = 1e3  # memory frequency
        self.entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)

        self.channel_qubits = 25
        self.init_fidelity = 0.99
        self.t_coherence = 0.01  # sec
        self.p_swap = 0.5

        self.total_distance = 150  # km

        self.n_runs = 10

        self.number_of_routers: int
        self.distance_proportion: str
        self.swapping_config: str


def compute_distances_distribution(end_to_end_distance: int, number_of_routers: int, distance_proportion: str) -> list[int]:
    """Computes the distribution of channel distances between nodes in a quantum or classical network.

    Args:
        end_to_end_distance (int): Total distance from source to destination.
        number_of_routers (int): Number of intermediate routers (excluding source and destination).
        distance_proportion (str): One of ['uniform', 'increasing', 'decreasing', 'mid_bottleneck'].

    Returns:
        List[int]: List of segment distances between nodes.

    """
    total_segments = number_of_routers + 1  # Source, routers, destination
    # Handle cases with no routers or just one router
    if number_of_routers == 0:
        return [end_to_end_distance]  # Entire distance as a single segment
    if distance_proportion == "uniform":
        return [end_to_end_distance // total_segments] * total_segments
    elif distance_proportion == "increasing":
        weights = [i * 2 + 1 for i in range(total_segments)]
        total_weight = sum(weights)
        distances = [end_to_end_distance * (w / total_weight) for w in weights]
        return [int(d) for d in distances]
    elif distance_proportion == "decreasing":
        weights = [i * 2 + 1 for i in range(total_segments)][::-1]
        total_weight = sum(weights)
        distances = [end_to_end_distance * (w / total_weight) for w in weights]
        return [int(d) for d in distances]
    if distance_proportion == "mid_bottleneck":
        # Compute base distance for edge segments
        edge_segments = total_segments - 2 if total_segments % 2 == 0 else total_segments - 1
        base_edge_distance = int(end_to_end_distance / (1.2 * edge_segments + (2 if total_segments % 2 == 0 else 1)))
        # Compute middle distances
        if total_segments % 2 == 0:  # Even segments: two middle segments
            middle_distance = int(base_edge_distance * 1.2)
            return (
                [base_edge_distance] * (edge_segments // 2)
                + [middle_distance, middle_distance]
                + [base_edge_distance] * (edge_segments // 2)
            )
        else:  # Odd segments: single middle segment
            middle_distance = int(base_edge_distance * 1.2)
            return [base_edge_distance] * (edge_segments // 2) + [middle_distance] + [base_edge_distance] * (edge_segments // 2)
    else:
        raise ValueError(f"Invalid distance proportion type: {distance_proportion}")


def generate_topology(p: ParameterSet) -> Topo:
    # Generate nodes
    nodes: list[TopoQNode] = []
    nodes.append(
        {
            "name": "S",
            "memory": {"decoherence_rate": 1 / p.t_coherence, "capacity": p.channel_qubits},
            "apps": [
                LinkLayer(
                    attempt_rate=p.entg_attempt_rate,
                    init_fidelity=p.init_fidelity,
                    alpha_db_per_km=p.fiber_alpha,
                    eta_d=p.eta_d,
                    eta_s=p.eta_s,
                    frequency=p.frequency,
                ),
                ProactiveForwarder(),
            ],
        }
    )
    for i in range(1, p.number_of_routers + 1):
        nodes.append(
            {
                "name": f"R{i}",
                "memory": {"decoherence_rate": 1 / p.t_coherence, "capacity": p.channel_qubits * 2},
                "apps": [
                    LinkLayer(
                        attempt_rate=p.entg_attempt_rate,
                        init_fidelity=p.init_fidelity,
                        alpha_db_per_km=p.fiber_alpha,
                        eta_d=p.eta_d,
                        eta_s=p.eta_s,
                        frequency=p.frequency,
                    ),
                    ProactiveForwarder(ps=p.p_swap),
                ],
            }
        )
    nodes.append(
        {
            "name": "D",
            "memory": {"decoherence_rate": 1 / p.t_coherence, "capacity": p.channel_qubits},
            "apps": [
                LinkLayer(
                    attempt_rate=p.entg_attempt_rate,
                    init_fidelity=p.init_fidelity,
                    alpha_db_per_km=p.fiber_alpha,
                    eta_d=p.eta_d,
                    eta_s=p.eta_s,
                    frequency=p.frequency,
                ),
                ProactiveForwarder(),
            ],
        }
    )

    # Compute distances
    distances = compute_distances_distribution(p.total_distance, p.number_of_routers, p.distance_proportion)

    # Generate qchannels and cchannels
    qchannels: list[TopoQChannel] = []
    cchannels: list[TopoCChannel] = []
    names = [node["name"] for node in nodes]
    for i, ch_len in enumerate(distances):
        qchannels.append(
            {
                "node1": names[i],
                "node2": names[i + 1],
                "capacity": p.channel_qubits,
                "parameters": {"length": ch_len},
            }
        )
        cchannels.append({"node1": names[i], "node2": names[i + 1], "parameters": {"length": ch_len}})

    # Add classical channels to controller
    for name in names:
        cchannels.append({"node1": "ctrl", "node2": name, "parameters": {"length": 1.0}})

    # Define controller
    controller: TopoController = {
        "name": "ctrl",
        "apps": [ProactiveRoutingControllerApp(routing_type="SRSP", swapping=p.swapping_config)],
    }

    return {"qnodes": nodes, "qchannels": qchannels, "cchannels": cchannels, "controller": controller}


def run_simulation(p: ParameterSet, seed: int) -> tuple[float, float]:
    json_topology = generate_topology(p)

    set_seed(seed)
    s = Simulator(0, p.sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topology = CustomTopology(json_topology)
    net = QuantumNetwork(topo=topology)

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
    p.swapping_config = f"swap_{num_routers}_{swap_conf}"
    if swap_conf == "vora":
        p.swapping_config += f"_{dist_prop}"

    entanglements = []
    expired = []
    for i in range(p.n_runs):
        print(f"Simulation: {num_routers} routers | {dist_prop} " + f"distances | {swap_conf} | run #{i + 1}")
        seed = p.seed_base + i
        e2e_count, expired_count = run_simulation(p, seed)
        # print(f"==> expired_count: {expired_count}")
        entanglements.append(e2e_count)
        expired.append(expired_count)

    mean_entg = np.mean(entanglements)
    std_entg = np.std(entanglements)
    mean_exp = np.mean(expired)
    std_exp = np.std(expired)

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
        workers: int = 1  # number of workers for parallel execution
        runs: int = 10  # number of trials per parameter set
        csv: str = ""  # save results as CSV file
        plt: str = ""  # save plot as image file

    args = Args().parse_args()

    p.n_runs = args.runs

    # Simulator loop with process-based parallelism
    with Pool(processes=args.workers) as pool:
        results = pool.starmap(run_row, itertools.product([p], NUM_ROUTERS_OPTIONS, DIST_PROPORTIONS, SWAP_CONFIGS))

    save_results(results, save_csv=args.csv, save_plt=args.plt)
