import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qns.network import QuantumNetwork, TimingModeEnum
from qns.network.protocol import LinkLayer, ProactiveForwarder, ProactiveRoutingControllerApp
from qns.network.route import DijkstraRouteAlgorithm
from qns.network.topology.customtopo import CustomTopology, Topo
from qns.simulator import Simulator
from qns.utils import log
from qns.utils.rnd import set_seed

log.set_default_level("DEBUG")

SEED_BASE = 100

# parameters
sim_duration = 3

fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)

channel_qubits = 1
init_fidelity = 0.99
p_swap = 0.5


# 3-nodes topology
swapping_config = "swap_1"
ch_1 = 32
ch_2 = 18


def generate_topology(t_coherence: float) -> Topo:
    """
    Defines the topology with globally declared simulation parameters.

    Args:
        key (float): memory coherence time in seconds.

    Returns:
        dict: the topology definition to be used to build the quantum network.
    """

    return {
        "qnodes": [
            {
                "name": "S",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": channel_qubits,
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
                    ProactiveForwarder(),
                ],
            },
            {
                "name": "R",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": channel_qubits * 2,
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
            },
            {
                "name": "D",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": channel_qubits,
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
                    ProactiveForwarder(),
                ],
            },
        ],
        "qchannels": [
            {
                "node1": "S",
                "node2": "R",
                "capacity": channel_qubits,
                "parameters": {"length": ch_1},
            },
            {
                "node1": "R",
                "node2": "D",
                "capacity": channel_qubits,
                "parameters": {"length": ch_2},
            },
        ],
        "cchannels": [
            {"node1": "S", "node2": "R", "parameters": {"length": ch_1}},
            {"node1": "R", "node2": "D", "parameters": {"length": ch_2}},
            {"node1": "ctrl", "node2": "S", "parameters": {"length": 1.0}},
            {"node1": "ctrl", "node2": "R", "parameters": {"length": 1.0}},
            {"node1": "ctrl", "node2": "D", "parameters": {"length": 1.0}},
        ],
        "controller": {"name": "ctrl", "apps": [ProactiveRoutingControllerApp(swapping=swapping_config)]},
    }


def run_simulation(t_coherence, seed):
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
    json_topology = generate_topology(t_coherence)

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

    e2e_rate = net.get_node("S").get_app(ProactiveForwarder).e2e_count / sim_duration

    return e2e_rate, total_decohered / total_etg if total_etg > 0 else 0


########################### Main #########################
results = {"T_cohere": [], "Mean Rate": [], "Std Rate": []}

# t_cohere_values = [2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 8e-2, 1e-1]
t_cohere_values = np.geomspace(2e-3, 1e-1, 8)

N_RUNS = 10
for t_cohere in t_cohere_values:
    rates = []
    for i in range(N_RUNS):
        print(f"T_cohere={t_cohere:.4f}, run {i + 1}")
        seed = SEED_BASE + i
        rate, *_ = run_simulation(t_cohere, seed)
        rates.append(rate)

    results["T_cohere"].append(t_cohere)
    results["Mean Rate"].append(np.mean(rates))
    results["Std Rate"].append(np.std(rates))

# Convert to DataFrame
df = pd.DataFrame(results)

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
plt.show()
