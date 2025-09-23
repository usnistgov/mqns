import json
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from mqns.entity import Controller
from mqns.network.network import QuantumNetwork
from mqns.network.proactive import (
    LinkLayer,
    MuxSchemeStatistical,
    ProactiveForwarder,
    ProactiveRoutingController,
    QubitAllocationType,
    RoutingPathSingle,
)
from mqns.network.topology import ClassicTopology, RandomTopology
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

"""
This script measures how simulation performance and outcomes scale as the
network size increases. A random topology is used with an average node degree of 3.5.
For each network size, the number of entanglement requests is chosen to be
proportional to the number of nodes, with 50% of nodes involved in src-dst requests.
Proactive forwarding is used with Statistical multiplexing and SWAP-ASAP swapping policy.

The simulation reports three key aspects across multiple trials:
  - Average end-to-end entanglement generation rate
  - Average fidelity of consumed entangled pairs
  - Simulator execution time
"""


# Command line arguments
class Args(Tap):
    runs: int = 3  # number of trials per parameter set
    json: str = ""  # save results as JSON file
    plt: str = ""  # save plot as image file


args = Args().parse_args()

log.set_default_level("CRITICAL")

# avg. node degree = 3.5
network_sizes: list[tuple[int, int]] = [
    (16, 28),
    (32, 56),
    (64, 112),
    (128, 224),
    (256, 448),
    (512, 896),
    (1024, 1792),
]
print(f"Simulate network sizes: {network_sizes}")


# parameters
sim_duration = 3

fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)

init_fidelity = 0.99
t_coherence = 5e-3  # 10e-3

p_swap = 0.5
swapping_policy = "asap"
nqubits = 50


def build_network(nnodes: int, nedges: int, nqubits: int) -> QuantumNetwork:
    """
    Defines the topology with globally declared simulation parameters.
    """

    topo = RandomTopology(
        nodes_number=nnodes,
        lines_number=nedges,
        qchannel_args={"length": 30},
        cchannel_args={"length": 30},
        memory_args={"capacity": nqubits, "decoherence_rate": 1 / t_coherence},
        nodes_apps=[
            LinkLayer(
                attempt_rate=entg_attempt_rate,
                init_fidelity=init_fidelity,
                alpha_db_per_km=fiber_alpha,
                eta_d=eta_d,
                eta_s=eta_s,
                frequency=frequency,
            ),
            ProactiveForwarder(ps=p_swap, mux=MuxSchemeStatistical()),
        ],
    )
    topo.controller = Controller("ctrl", apps=[ProactiveRoutingController()])

    # Default: Dijkstra with hop count metric
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)

    topo.connect_controller(net.nodes)

    for qchannel in net.qchannels:
        qchannel.assign_memory_qubits(capacity=3)

    return net


def run_simulation(nnodes: int, nedges: int, nqubits: int, seed: int):
    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    net = build_network(nnodes, nedges, nqubits)
    net.install(s)

    # select random S-D pairs
    ctrl = net.get_controller().get_app(ProactiveRoutingController)

    # number of requests is proportional to network size
    num_requests = max(2, int(nnodes / 4))

    # Time to generate requests
    t0 = time.perf_counter()
    net.random_requests(num_requests, min_hops=2, max_hops=5)
    t1 = time.perf_counter()
    print(f"Random requests generation took {t1 - t0:.6f} seconds ({num_requests}) requests")

    for req in net.requests:
        ctrl.install_path(
            RoutingPathSingle(req.src.name, req.dst.name, qubit_allocation=QubitAllocationType.DISABLED, swap="asap")
        )

    s.run()

    #### get stats: e2e_rate and mean_fidelity
    stats = []
    for req in net.requests:
        fw = req.src.get_app(ProactiveForwarder)
        stats.append((fw.cnt.n_consumed / sim_duration, fw.cnt.consumed_avg_fidelity))

    # [(path 1), (path 2), ...]
    return stats, s.time_spend


# Simulation constants
SEED_BASE = 200

# Collect (rate, fid) for all paths and execution time over all runs
results = {netsize: [] for netsize in network_sizes}
time_results = {netsize: [] for netsize in network_sizes}

# Run simulation
for nnodes, nedges in network_sizes:
    for i in range(args.runs):
        print(f"network size: {(nnodes, nedges)}, run #{i}")
        seed = SEED_BASE + i

        # [(path 1 rate, path 1 fid), (path 2 rate, path 2 fid), ...]
        path_stats, time_spent = run_simulation(
            nnodes=nnodes,
            nedges=nedges,
            nqubits=nqubits,
            seed=seed,
        )
        results[(nnodes, nedges)].extend(path_stats)
        time_results[(nnodes, nedges)].append(float(time_spent))

# Optional: save per-path results
if args.json:
    with open(args.json, "w") as file:
        # make keys serializable and include times
        json.dump(
            {
                "results": {str(k): v for k, v in results.items()},
                "time_results": {str(k): v for k, v in time_results.items()},
            },
            file,
        )

# =========================
# Plot results
# =========================
mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "errorbar.capsize": 4,
    }
)

x_ticks = list(range(len(network_sizes)))
x_ticklabels = [f"({n},{e})" for (n, e) in network_sizes]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
ax_rate, ax_fid, ax_time = axes

# ---- Collect means and stds
mean_rates, std_rates = [], []
mean_fids, std_fids = [], []
mean_times, std_times = [], []

for netsize in network_sizes:
    data = results[netsize]
    if data:
        rates_arr = np.array([r for (r, _) in data], dtype=float)
        fids_arr = np.array([f for (_, f) in data], dtype=float)
        times = np.array(time_results[netsize], dtype=float)

        mean_rates.append(rates_arr.mean())
        std_rates.append(rates_arr.std(ddof=1) if rates_arr.size > 1 else 0.0)

        mean_fids.append(fids_arr.mean())
        std_fids.append(fids_arr.std(ddof=1) if fids_arr.size > 1 else 0.0)

        mean_times.append(times.mean())
        std_times.append(times.std(ddof=1) if times.size > 1 else 0.0)
    else:
        mean_rates.append(np.nan)
        std_rates.append(0.0)
        mean_fids.append(np.nan)
        std_fids.append(0.0)
        mean_times.append(np.nan)
        std_times.append(0.0)

# ---- Subplot 1: Entanglement Rate
ax_rate.errorbar(x_ticks, mean_rates, yerr=std_rates, marker=None, linestyle="-", label="Rate")
ax_rate.set_title("Average Entanglement Rate")
ax_rate.set_ylabel("E2E Rate (eps)")
ax_rate.set_xlabel("Net. size")
ax_rate.set_xticks(x_ticks, x_ticklabels, rotation=45, ha="right")
ax_rate.grid(True, alpha=0.4)

# ---- Subplot 2: Fidelity
ax_fid.errorbar(x_ticks, mean_fids, yerr=std_fids, marker=None, linestyle="-", label="Fidelity")
ax_fid.set_title("Average Fidelity")
ax_fid.set_ylabel("Fidelity")
ax_fid.set_xlabel("Net. size")
ax_fid.set_xticks(x_ticks, x_ticklabels, rotation=45, ha="right")
ax_fid.grid(True, alpha=0.4)

# ---- Subplot 3: Simulation Execution Time
ax_time.errorbar(x_ticks, mean_times, yerr=std_times, marker=None, linestyle="-", label="Time")
ax_time.set_title("Average Execution Time")
ax_time.set_ylabel("Time (s)")
ax_time.set_xlabel("Net. size")
ax_time.set_xticks(x_ticks, x_ticklabels, rotation=45, ha="right")
ax_time.grid(True, alpha=0.4)

plt.show()

# Optional: save figures
if args.plt:
    base, ext = (args.plt.rsplit(".", 1) + ["png"])[:2]
    fig.savefig(f"{base}_summary.{ext}", dpi=300, transparent=True)
