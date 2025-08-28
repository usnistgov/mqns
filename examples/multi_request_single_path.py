import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from qns.entity.node import Application
from qns.network.network import QuantumNetwork
from qns.network.proactive import (
    LinkLayer,
    MuxScheme,
    MuxSchemeDynamicEpr,
    MuxSchemeStatistical,
    ProactiveForwarder,
    ProactiveRoutingController,
    QubitAllocationType,
    RoutingPathSingle,
    select_weighted_by_swaps,
)
from qns.network.topology.customtopo import CustomTopology, Topo
from qns.simulator import Simulator
from qns.utils import log, set_seed


# Command line arguments
class Args(Tap):
    runs: int = 3  # number of trials per parameter set
    json: str = ""  # save results as JSON file
    plt: str = ""  # save plot as image file


args = Args().parse_args()

log.set_default_level("CRITICAL")

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

# Quantum channel lengths
ch_S1_R1 = 10
ch_R1_R2 = 10
ch_R2_R3 = 10
ch_R3_D1 = 10
ch_S2_R2 = 10
ch_R3_D2 = 10


def generate_topology(t_coherence: float, p_swap: float, mux: MuxScheme) -> Topo:
    """
    Defines the topology with globally declared simulation parameters.
    """

    def make_qnode_apps() -> list[Application]:
        return [
            LinkLayer(
                attempt_rate=entg_attempt_rate,
                init_fidelity=init_fidelity,
                alpha_db_per_km=fiber_alpha,
                eta_d=eta_d,
                eta_s=eta_s,
                frequency=frequency,
            ),
            ProactiveForwarder(ps=p_swap, mux=mux),
        ]

    return {
        "qnodes": [
            {
                "name": "S1",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
                },
                "apps": make_qnode_apps(),
            },
            {
                "name": "S2",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
                },
                "apps": make_qnode_apps(),
            },
            {
                "name": "D1",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
                },
                "apps": make_qnode_apps(),
            },
            {
                "name": "D2",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
                },
                "apps": make_qnode_apps(),
            },
            {
                "name": "R1",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 2,
                },
                "apps": make_qnode_apps(),
            },
            {
                "name": "R2",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 3,
                },
                "apps": make_qnode_apps(),
            },
            {
                "name": "R3",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 3,
                },
                "apps": make_qnode_apps(),
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
            "apps": [
                ProactiveRoutingController(
                    [
                        RoutingPathSingle("S1", "D1", qubit_allocation=QubitAllocationType.DISABLED, swap=swapping_policy),
                        RoutingPathSingle("S2", "D2", qubit_allocation=QubitAllocationType.DISABLED, swap=swapping_policy),
                    ]
                )
            ],
        },
    }


def run_simulation(t_coherence: float, p_swap: float, mux: MuxScheme, seed: int):
    json_topology = generate_topology(t_coherence, p_swap, mux)
    # print(json_topology)

    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = CustomTopology(json_topology)
    net = QuantumNetwork(topo=topo)
    net.install(s)

    s.run()

    #### get stats: e2e_rate and mean_fidelity
    fw_s1 = net.get_node("S1").get_app(ProactiveForwarder)
    fw_s2 = net.get_node("S2").get_app(ProactiveForwarder)
    # [(path 1), (path 2), ...]
    return [
        (fw_s1.cnt.n_consumed / sim_duration, fw_s1.cnt.consumed_avg_fidelity),
        (fw_s2.cnt.n_consumed / sim_duration, fw_s2.cnt.consumed_avg_fidelity),
    ]


# Simulation constants
SEED_BASE = 100
p_swap = 0.5
t_cohere_values = [5e-3, 10e-3, 20e-3]

# Strategy configs
strategies: dict[str, MuxScheme] = {
    "Statistical Mux.": MuxSchemeStatistical(),
    "Random Alloc.": MuxSchemeDynamicEpr(),
    "Swap-weighted Alloc.": MuxSchemeDynamicEpr(path_select_fn=select_weighted_by_swaps),
}


results = {strategy: {0: [], 1: []} for strategy in strategies}

# Run simulation
for strategy, mux in strategies.items():
    for t_cohere in t_cohere_values:
        path_rates: list[list[float]] = [[], []]
        path_fids: list[list[float]] = [[], []]
        for i in range(args.runs):
            print(f"{strategy}, T_cohere={t_cohere:.3f}, run #{i}")
            seed = SEED_BASE + i
            (rate1, fid1), (rate2, fid2) = run_simulation(
                t_coherence=t_cohere,
                p_swap=p_swap,
                mux=mux,
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


if args.json:
    with open(args.json, "w") as file:
        json.dump(results, file)

mpl.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 22,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "errorbar.capsize": 4,
    }
)

fig, axs = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey="row")

# Plot Entanglement Rate
for strategy in strategies:
    for path in [0, 1]:
        rates = [results[strategy][path][i][0] for i in range(len(t_cohere_values))]
        stds = [results[strategy][path][i][1] for i in range(len(t_cohere_values))]
        axs[0][path].errorbar(
            [t * 1e3 for t in t_cohere_values],
            rates,
            yerr=stds,
            marker="o",
            label=strategy,
        )

axs[0][0].set_title("S1-D1")
axs[0][1].set_title("S2-D2")
for ax in axs[0]:
    ax.set_ylabel("E2E Rate (eps)")
    ax.grid(True)

# Plot Fidelity
for strategy in strategies:
    for path in [0, 1]:
        fids = [results[strategy][path][i][2] for i in range(len(t_cohere_values))]
        stds = [results[strategy][path][i][3] for i in range(len(t_cohere_values))]
        axs[1][path].errorbar(
            [t * 1e3 for t in t_cohere_values],
            fids,
            yerr=stds,
            marker="s",
            label=strategy,
        )

axs[1][0].set_title("S1-D1")
axs[1][1].set_title("S2-D2")
for ax in axs[1]:
    ax.set_xlabel("T_cohere (ms)")
    ax.set_ylabel("Fidelity")
    ax.grid(True)

axs[1][1].legend(title="Strategy", loc="lower right")
fig.tight_layout(rect=(0, 0, 1, 0.95))
if args.plt:
    plt.savefig(args.plt, dpi=300, transparent=True)
plt.show()
