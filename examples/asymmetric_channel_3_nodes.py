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

swapping_config = "swap_1"


def generate_topology(
    nodes: list[str],
    mem_capacities: list[int],
    channel_lengths: list[float],
    capacities: list[tuple[int, int]],
    t_coherence: float,
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
    controller = {"name": "ctrl", "apps": [ProactiveRoutingControllerApp(routing_type="SRSP", swapping=swapping_config)]}
    for node in nodes:
        cchannels.append({"node1": "ctrl", "node2": node, "parameters": {"length": 1.0, "delay": 1.0 / light_speed}})

    return {"qnodes": qnodes, "qchannels": qchannels, "cchannels": cchannels, "controller": controller}


def run_simulation(
    nodes: list[str],
    mem_capacities: list[int],
    channel_lengths: list[float],
    capacities: list[tuple[int, int]],
    t_coherence: float,
    seed: int,
):
    json_topology = generate_topology(nodes, mem_capacities, channel_lengths, ch_capacities, t_coherence)
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

# Constants
N_RUNS = 3
SEED_BASE = 42
TOTAL_QUBITS = 6

# Experiment parameters
t_cohere_values = [5e-3, 10e-3, 20e-3]
mem_allocs = [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]
mem_labels = [str(m) for m in mem_allocs]

channel_configs = {
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
            for i in range(N_RUNS):
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
plt.show()
