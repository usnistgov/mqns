import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qns.network import QuantumNetwork, TimingModeEnum
from qns.network.protocol.link_layer import LinkLayer
from qns.network.protocol.proactive_forwarder import ProactiveForwarder
from qns.network.protocol.proactive_routing_controller import ProactiveRoutingControllerApp
from qns.network.route.dijkstra import DijkstraRouteAlgorithm
from qns.network.topology.customtopo import CustomTopology
from qns.simulator.simulator import Simulator
from qns.utils import log
from qns.utils.rnd import set_seed

log.logger.setLevel(logging.CRITICAL)

SEED_BASE = 100

light_speed = 2 * 10**5 # km/s


# parameters
sim_duration = 5      # 9

fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e3                  # memory frequency
entg_attempt_rate = 50e6         # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)

channel_qubits = 25
init_fidelity = 0.99
t_coherence = 0.01    # sec
p_swap = 0.5


def compute_distances_distribution(end_to_end_distance, number_of_routers, distance_proportion):
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
        weights = [i*2+ 1 for i in range(total_segments)]
        total_weight = sum(weights)
        distances = [end_to_end_distance * (w / total_weight) for w in weights]
        return [int(d) for d in distances]
    elif distance_proportion == "decreasing":
        weights = [i*2+ 1 for i in range(total_segments)][::-1]
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
            return [base_edge_distance] * (edge_segments // 2) + [middle_distance, middle_distance] + [base_edge_distance] \
                * (edge_segments // 2)
        else:  # Odd segments: single middle segment
            middle_distance = int(base_edge_distance * 1.2)
            return [base_edge_distance] * (edge_segments // 2) + [middle_distance] + [base_edge_distance] * (edge_segments // 2)
    else:
        raise ValueError(f"Invalid distance proportion type: {distance_proportion}")

def generate_topology(number_of_routers, distance_proportion, swapping_config, total_distance):
    # Generate nodes
    nodes = [{"name": "S", "memory": {"decoherence_rate": 1 / t_coherence, "capacity": channel_qubits},
              "apps": [LinkLayer(attempt_rate=entg_attempt_rate, init_fidelity=init_fidelity,
                                 alpha_db_per_km=fiber_alpha,
                                 eta_d=eta_d, eta_s=eta_s,
                                 frequency=frequency),
                       ProactiveForwarder()]}]
    for i in range(1, number_of_routers + 1):
        nodes.append({
            "name": f"R{i}",
            "memory": {"decoherence_rate": 1 / t_coherence, "capacity": channel_qubits * 2},
            "apps": [LinkLayer(attempt_rate=entg_attempt_rate, init_fidelity=init_fidelity,
                               alpha_db_per_km=fiber_alpha,
                               eta_d=eta_d, eta_s=eta_s,
                               frequency=frequency), ProactiveForwarder(ps=p_swap)]
        })
    nodes.append({"name": "D", "memory": {"decoherence_rate": 1 / t_coherence, "capacity": channel_qubits},
                  "apps": [LinkLayer(attempt_rate=entg_attempt_rate, init_fidelity=init_fidelity,
                                     alpha_db_per_km=fiber_alpha,
                                     eta_d=eta_d, eta_s=eta_s,
                                     frequency=frequency), ProactiveForwarder()]})

    # Compute distances
    distances = compute_distances_distribution(total_distance, number_of_routers, distance_proportion)

    # Generate qchannels and cchannels
    qchannels = []
    cchannels = []
    names = ["S"] + [f"R{i}" for i in range(1, number_of_routers + 1)] + ["D"]
    for i in range(len(names) - 1):
        ch_len = distances[i]
        qchannels.append({
            "node1": names[i],
            "node2": names[i+1],
            "capacity": channel_qubits,
            "parameters": {
                "length": ch_len,
                "delay": ch_len / light_speed
            }
        })
        cchannels.append({
            "node1": names[i],
            "node2": names[i+1],
            "parameters": {
                "length": ch_len,
                "delay": ch_len / light_speed
            }
        })

    # Add classical channels to controller
    for name in names:
        cchannels.append({
            "node1": "ctrl",
            "node2": name,
            "parameters": {"length": 1.0, "delay": 1.0 / light_speed}
        })

    # Define controller
    controller = {
        "name": "ctrl",
        "apps": [ProactiveRoutingControllerApp(swapping=swapping_config)]
    }

    return {
        "qnodes": nodes,
        "qchannels": qchannels,
        "cchannels": cchannels,
        "controller": controller
    }


def run_simulation(number_of_routers, distance_proportion, swapping_config, total_distance, seed):
    json_topology = generate_topology(number_of_routers, distance_proportion, swapping_config, total_distance)

    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topology = CustomTopology(json_topology)
    net = QuantumNetwork(
        topo=topology,
        route=DijkstraRouteAlgorithm(),
        timing_mode=TimingModeEnum.ASYNC,
        t_slot=0,
        t_ext=0,
        t_int=0
    )

    sim_run = sim_duration
    net.install(s)
    s.run()

    #### get stats
    # total_etg = 0
    total_decohered = 0
    for node in net.get_nodes():
        ll_app = node.get_apps(LinkLayer)[0]
        # total_etg+=ll_app.etg_count
        total_decohered+=ll_app.decoh_count
    e2e_count = net.get_node("S").get_apps(ProactiveForwarder)[0].e2e_count

    return e2e_count / sim_run, total_decohered / e2e_count if e2e_count > 0 else 0


# Configuration parameters
TOTAL_DISTANCE = 150  # km

N_RUNS = 10     # 30
NUM_ROUTERS_OPTIONS = [3, 4, 5]
DIST_PROPORTIONS = ["decreasing", "increasing", "mid_bottleneck", "uniform"]
SWAP_CONFIGS = ["asap", "baln", "vora", "l2r"]

results = []

# Simulation loop
for num_routers, dist_prop, swap_conf in itertools.product(NUM_ROUTERS_OPTIONS, DIST_PROPORTIONS, SWAP_CONFIGS):
    full_swapping_config = f"swap_{num_routers}_{swap_conf}"
    if swap_conf == "vora":
        full_swapping_config+=f"_{dist_prop}"
    entanglements = []
    expired = []
    for i in range(N_RUNS):
        print(f"Simulation: {num_routers} routers | {dist_prop} "+
          f"distances | {swap_conf} | run #{i+1}")
        seed = SEED_BASE + i
        e2e_count, expired_count = run_simulation(num_routers, dist_prop, full_swapping_config, TOTAL_DISTANCE, seed)
        # print(f"==> expired_count: {expired_count}")
        entanglements.append(e2e_count)
        expired.append(expired_count)

    mean_entg = np.mean(entanglements)
    std_entg = np.std(entanglements)
    mean_exp = np.mean(expired)
    std_exp = np.std(expired)

    results.append({
        "Routers": num_routers,
        "Distance Distribution": dist_prop,
        "Swapping Config": swap_conf,
        "Entanglements Per Second": mean_entg,
        "Entanglements Std": std_entg,
        "Entanglements All Runs": entanglements,
        "Expired Memories Per Entanglement": mean_exp,
        "Expired Memories Std": std_exp,
        "Expired Memories All Runs": expired
    })

df = pd.DataFrame(results)
# df.to_csv("trend_results_3.csv", index=False)

# === Combined Plot ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey="row")

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
            row = df_subset[(df_subset["Distance Distribution"] == dist_prop) &
                            (df_subset["Swapping Config"] == swap_conf)]
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
            row = df_subset[(df_subset["Distance Distribution"] == dist_prop) &
                            (df_subset["Swapping Config"] == swap_conf)]
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
plt.show()
