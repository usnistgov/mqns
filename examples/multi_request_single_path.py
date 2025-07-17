import logging
import random
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from qns.network import QuantumNetwork, TimingModeEnum
from qns.network.protocol.fib import FIBEntry
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

# parameters
sim_duration = 3

fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)

init_fidelity = 0.99

swapping_policy = "asap"

# Multipath settings
routing_type = "MRSP_DYNAMIC"  # Controller installs one path for each S-D request, without qubit-path allocation

# Quantum channel lengths
ch_S1_R1 = 10
ch_R1_R2 = 10
ch_R2_R3 = 10
ch_R3_D1 = 10
ch_S2_R2 = 25
ch_R3_D2 = 15


# path selection strategy for dynamic EPR allocation
def select_weighted_by_swaps(fibs: list[FIBEntry]) -> int:
    if not fibs:
        return None

    # Lower swaps = higher weight
    weights = [1.0 / (1 + len(e["swap_sequence"])) for e in fibs]
    total = sum(weights)
    probabilities = [w / total for w in weights]
    return random.choices(fibs, weights=probabilities, k=1)[0]["path_id"]


def generate_topology(
    t_coherence: float, p_swap: float, statistical_mux: bool, path_select_fn: Callable[[list[FIBEntry]], int] | None
) -> dict:
    """
    Defines the topology with globally declared simulation parameters.

    Returns:
        dict: the topology definition to be used to build the quantum network.
    """
    return {
        "qnodes": [
            {
                "name": "S1",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux, path_select_fn=path_select_fn),
                ],
            },
            {
                "name": "S2",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux, path_select_fn=path_select_fn),
                ],
            },
            {
                "name": "D1",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux, path_select_fn=path_select_fn),
                ],
            },
            {
                "name": "D2",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux, path_select_fn=path_select_fn),
                ],
            },
            {
                "name": "R1",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 2,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux, path_select_fn=path_select_fn),
                ],
            },
            {
                "name": "R2",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 3,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux, path_select_fn=path_select_fn),
                ],
            },
            {
                "name": "R3",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 3,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux, path_select_fn=path_select_fn),
                ],
            },
        ],
        "qchannels": [
            {"node1": "S1", "node2": "R1", "capacity": 1, "parameters": {"length": ch_S1_R1}},
            {"node1": "R1", "node2": "R2", "capacity": 1, "parameters": {"length": ch_R1_R2}},
            {"node1": "R2", "node2": "R3", "capacity": 1, "parameters": {"length": ch_R2_R3}},
            {"node1": "R3", "node2": "D1", "capacity": 1, "parameters": {"length": ch_R3_D1}},
            {"node1": "S2", "node2": "R2", "capacity": 1, "parameters": {"length": ch_S2_R2}},
            {"node1": "R3", "node2": "D2", "capacity": 1, "parameters": {"length": ch_R3_D2}},
        ],
        "cchannels": [
            {"node1": "S1", "node2": "R1", "parameters": {"length": ch_S1_R1}},
            {"node1": "R1", "node2": "R2", "parameters": {"length": ch_R1_R2}},
            {"node1": "R2", "node2": "R3", "parameters": {"length": ch_R2_R3}},
            {"node1": "R3", "node2": "D1", "parameters": {"length": ch_R3_D1}},
            {"node1": "S2", "node2": "R2", "parameters": {"length": ch_S2_R2}},
            {"node1": "R3", "node2": "D2", "parameters": {"length": ch_R3_D2}},
            {"node1": "ctrl", "node2": "S1", "parameters": {"length": 1.0}},
            {"node1": "ctrl", "node2": "S2", "parameters": {"length": 1.0}},
            {"node1": "ctrl", "node2": "R1", "parameters": {"length": 1.0}},
            {"node1": "ctrl", "node2": "R2", "parameters": {"length": 1.0}},
            {"node1": "ctrl", "node2": "R3", "parameters": {"length": 1.0}},
            {"node1": "ctrl", "node2": "D1", "parameters": {"length": 1.0}},
            {"node1": "ctrl", "node2": "D2", "parameters": {"length": 1.0}},
        ],
        "controller": {
            "name": "ctrl",
            "apps": [ProactiveRoutingControllerApp(swapping_policy=swapping_policy, routing_type=routing_type)],
        },
    }


def run_simulation(
    t_coherence: float, p_swap: float, statistical_mux: bool, path_select_fn: Callable[[list[FIBEntry]], int] | None, seed: int
):
    json_topology = generate_topology(t_coherence, p_swap, statistical_mux, path_select_fn)
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
        ll_app = node.get_apps(LinkLayer)[0]
        total_etg += ll_app.etg_count
        total_decohered += ll_app.decoh_count

    e2e_rate_1 = net.get_node("S1").get_apps(ProactiveForwarder)[0].e2e_count / sim_duration
    e2e_rate_2 = net.get_node("S2").get_apps(ProactiveForwarder)[0].e2e_count / sim_duration

    print(f"E2E etg rate [S1-D1]: {e2e_rate_1}")
    print(f"E2E etg rate [S2-D2]: {e2e_rate_2}")
    print(f"Expired memories: {total_decohered / total_etg if total_etg > 0 else 0}")

    #### get stats
    e2e_count_1 = net.get_node("S1").get_app(ProactiveForwarder).e2e_count
    e2e_rate_1 = e2e_count_1 / sim_duration
    mean_fidelity_1 = net.get_node("S1").get_app(ProactiveForwarder).fidelity / e2e_count_1 if e2e_count_1 > 0 else 0

    e2e_count_2 = net.get_node("S2").get_app(ProactiveForwarder).e2e_count
    e2e_rate_2 = e2e_count_2 / sim_duration
    mean_fidelity_2 = net.get_node("S2").get_app(ProactiveForwarder).fidelity / e2e_count_2 if e2e_count_2 > 0 else 0

    # [(path 1), (path 2), ...]
    return [(e2e_rate_1, mean_fidelity_1), (e2e_rate_2, mean_fidelity_2)]


# Simulation constants
N_RUNS = 3
SEED_BASE = 100
p_swap = 0.5
t_cohere_values = [5e-3, 10e-3, 20e-3]

# Strategy configs
strategies = {
    "Statistical Mux": {"statistical_mux": True, "select_fn": None},
    "Random Selection": {"statistical_mux": False, "select_fn": None},
    "Custom Selection": {"statistical_mux": False, "select_fn": select_weighted_by_swaps},
}

# Results: strategy -> path -> t_cohere -> metrics
results = {strategy: {0: [], 1: []} for strategy in strategies}

# Run simulation
for strategy, config in strategies.items():
    for t_cohere in t_cohere_values:
        path_rates = [[], []]
        path_fids = [[], []]
        for i in range(N_RUNS):
            print(f"{strategy, config}, T_cohere={t_cohere:.3f}, run #{i}")
            seed = SEED_BASE + i
            (rate1, fid1), (rate2, fid2) = run_simulation(
                t_coherence=t_cohere,
                p_swap=p_swap,
                statistical_mux=config["statistical_mux"],
                path_select_fn=config["select_fn"],
                seed=seed,
            )
            path_rates[0].append(rate1)
            path_rates[1].append(rate2)
            path_fids[0].append(fid1)
            path_fids[1].append(fid2)
        for path in [0, 1]:
            mean_rate = np.mean(path_rates[path])
            std_rate = np.std(path_rates[path])
            mean_fid = np.mean(path_fids[path])
            std_fid = np.std(path_fids[path])
            results[strategy][path].append((mean_rate, std_rate, mean_fid, std_fid))

# Plot Entanglement Rate
fig_rate, axs_rate = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for strategy in strategies:
    for path in [0, 1]:
        rates = [results[strategy][path][i][0] for i in range(len(t_cohere_values))]
        stds = [results[strategy][path][i][1] for i in range(len(t_cohere_values))]
        axs_rate[path].errorbar([t * 1e3 for t in t_cohere_values], rates, yerr=stds, marker="o", label=strategy, capsize=3)

axs_rate[0].set_title("S1-D1 Entanglement Rate")
axs_rate[1].set_title("S2-D2 Entanglement Rate")
for ax in axs_rate:
    ax.set_xlabel("T_cohere (ms)")
    ax.set_ylabel("Entanglement Rate (eps)")
    ax.grid(True)
axs_rate[-1].legend()
fig_rate.suptitle("Entanglement Rate vs Coherence Time")
fig_rate.tight_layout()
plt.show()

# Plot Fidelity
fig_fid, axs_fid = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for strategy in strategies:
    for path in [0, 1]:
        fids = [results[strategy][path][i][2] for i in range(len(t_cohere_values))]
        stds = [results[strategy][path][i][3] for i in range(len(t_cohere_values))]
        axs_fid[path].errorbar([t * 1e3 for t in t_cohere_values], fids, yerr=stds, marker="s", label=strategy, capsize=3)

axs_fid[0].set_title("S1-D1 Fidelity")
axs_fid[1].set_title("S2-D2 Fidelity")
for ax in axs_fid:
    ax.set_xlabel("T_cohere (ms)")
    ax.set_ylabel("Fidelity")
    ax.grid(True)
axs_fid[-1].legend()
fig_fid.suptitle("Fidelity vs Coherence Time")
fig_fid.tight_layout()
plt.show()
