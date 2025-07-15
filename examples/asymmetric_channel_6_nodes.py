import logging

import matplotlib.pyplot as plt
import numpy as np

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

light_speed = 2 * 10**5  # km/s

# parameters
sim_duration = 3

fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)

init_fidelity = 0.99
p_swap = 0.5


def generate_topology(
    nodes: list[str],
    mem_capacities: list[int],
    channel_lengths: list[float],
    capacities: list[tuple[int, int]],
    t_coherence: float,
    swapping_order: str,
) -> dict:
    """
    Generate a linear topology with explicit memory and channel configurations.

    Args:
        nodes (list[str]): List of node names.
        mem_capacities (list[int]): Number of qubits per node.
        channel_lengths (list[float]): Lengths of quantum channels between adjacent nodes.
        capacities (list[tuple[int, int]]): (left, right) qubit allocation per qchannel.

    Returns:
        dict: A topology dictionary.
    """
    if len(nodes) != len(mem_capacities):
        raise ValueError("mem_capacities must match number of nodes")
    if len(channel_lengths) != len(nodes) - 1:
        raise ValueError("channel_lengths must be len(nodes) - 1")
    if len(capacities) != len(nodes) - 1:
        raise ValueError("capacities must be len(nodes) - 1")

    # Create QNodes
    qnodes = []
    for i, name in enumerate(nodes):
        qnodes.append(
            {
                "name": name,
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": mem_capacities[i],
                },
                "apps": [
                    LinkLayer(
                        attempt_rate=entg_attempt_rate,
                        init_fidelity=init_fidelity,
                        alpha_db_per_km=fiber_alpha,
                        eta_d=eta_d,
                        eta_s=eta_s,
                        frequency=frequency,
                    ),
                    ProactiveForwarder(ps=p_swap),
                ],
            }
        )

    # Create Quantum Channels
    qchannels = []
    for i in range(len(nodes) - 1):
        node1, node2 = nodes[i], nodes[i + 1]
        length = channel_lengths[i]
        cap1, cap2 = capacities[i]

        qchannels.append(
            {
                "node1": node1,
                "node2": node2,
                "capacity1": cap1,
                "capacity2": cap2,
                "parameters": {"length": length, "delay": length / light_speed},
            }
        )

    # Classical Channels
    cchannels = []
    for i in range(len(nodes) - 1):
        node1, node2 = nodes[i], nodes[i + 1]
        length = channel_lengths[i]
        cchannels.append({"node1": node1, "node2": node2, "parameters": {"length": length, "delay": length / light_speed}})

    # Controller and links to all nodes
    controller = {"name": "ctrl", "apps": [ProactiveRoutingControllerApp(routing_type="SRSP", swapping=swapping_order)]}
    for node in nodes:
        cchannels.append({"node1": "ctrl", "node2": node, "parameters": {"length": 1.0, "delay": 1.0 / light_speed}})

    return {"qnodes": qnodes, "qchannels": qchannels, "cchannels": cchannels, "controller": controller}


def run_simulation(
    nodes: list[str],
    mem_capacities: list[int],
    channel_lengths: list[float],
    capacities: list[tuple[int, int]],
    t_coherence: float,
    swapping_order: str,
    seed: int,
):
    json_topology = generate_topology(nodes, mem_capacities, channel_lengths, ch_capacities, t_coherence, swapping_order)
    # print(json_topology)

    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = CustomTopology(json_topology)
    net = QuantumNetwork(topo=topo, route=DijkstraRouteAlgorithm(), timing_mode=TimingModeEnum.ASYNC)
    net.install(s)

    s.run()

    #### get stats
    total_etg = 0
    total_decohered = 0
    for node in net.get_nodes():
        ll_app = node.get_app(LinkLayer)
        total_etg += ll_app.etg_count
        total_decohered += ll_app.decoh_count

    e2e_count = net.get_node("S").get_app(ProactiveForwarder).e2e_count
    e2e_rate = e2e_count / sim_duration
    mean_fidelity = net.get_node("S").get_app(ProactiveForwarder).fidelity / e2e_count if e2e_count > 0 else 0

    return e2e_rate, total_decohered / total_etg if total_etg > 0 else 0, mean_fidelity


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
channel_lengths = [32, 18, 35, 16, 24]
TOTAL_QUBITS = 6

N_RUNS = 3
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
            for i in range(N_RUNS):
                seed = SEED_BASE + i
                swapping_config = order

                ch_capacities = mem_allocs

                rate, *_ = run_simulation(
                    nodes=nodes,
                    mem_capacities=mem_capacities,
                    channel_lengths=channel_lengths,
                    capacities=ch_capacities,
                    t_coherence=t_cohere,
                    swapping_order=swapping_config,
                    seed=seed,
                )
                run_rates.append(rate)

            results[mem_label][order][t_cohere] = (np.mean(run_rates), np.std(run_rates))


fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

colors = {"baln": "green", "baln2": "red", "l2r": "purple", "r2l": "brown", "asap": "pink"}
markers = {"baln": "v", "baln2": "s", "l2r": "^", "r2l": "x", "asap": "d"}

for ax_idx, (mem_label, policy_dict) in enumerate(results.items()):
    ax = axs[ax_idx]
    for full_policy, t_dict in policy_dict.items():
        policy = full_policy.replace("swap_4_", "")  # Extract "baln", "r2l", etc.
        means = [t_dict[t][0] for t in t_cohere_values]
        stds = [t_dict[t][1] for t in t_cohere_values]
        ax.errorbar(
            [t * 1e3 for t in t_cohere_values],
            means,
            yerr=stds,
            fmt=markers[policy],
            linestyle="-",
            capsize=3,
            label=policy,
            color=colors[policy],
        )
    ax.set_title(f"Mem. Alloc: {mem_label}")
    ax.set_xlabel("T_cohere (ms)")
    if ax_idx == 0:
        ax.set_ylabel("Throughput (eps)")
    ax.grid(True, linestyle="--")

axs[-1].legend(title="Policy")
plt.tight_layout()
plt.show()
