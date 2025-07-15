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
sim_duration = 5

fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)

channel_qubits = 2
init_fidelity = 0.7
p_swap = 0.5


# 3-nodes topology
swapping_config = "swap_1"
ch_1 = 32
ch_2 = 18


def generate_topology(t_coherence) -> Topo:
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

    cnt = net.get_node("S").get_app(ProactiveForwarder).cnt
    return cnt.n_consumed / sim_duration, cnt.consumed_avg_fidelity, total_decohered / total_etg if total_etg > 0 else 0


results = {"T_cohere": [], "Mean Rate": [], "Std Rate": [], "Mean F": [], "Std F": []}

t_cohere_values = [1]
# t_cohere_values = [2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 8e-2, 1e-1]
# t_cohere_values = np.geomspace(2e-3, 1e-1, 8)

N_RUNS = 1
for t_cohere in t_cohere_values:
    rates = []
    fids = []
    for i in range(N_RUNS):
        print(f"T_cohere={t_cohere:.4f}, run {i + 1}")
        seed = SEED_BASE + i
        rate, f, *_ = run_simulation(t_cohere, seed)
        rates.append(rate)
        fids.append(f)

    results["T_cohere"].append(t_cohere)
    results["Mean Rate"].append(np.mean(rates))
    results["Std Rate"].append(np.std(rates))
    results["Mean F"].append(np.mean(fids))
    results["Std F"].append(np.std(fids))

# Convert to DataFrame
df = pd.DataFrame(results)

print(df)
