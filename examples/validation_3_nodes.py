import logging

from qns.network.route.dijkstra import DijkstraRouteAlgorithm
from qns.simulator.simulator import Simulator
from qns.network import QuantumNetwork, TimingModeEnum
import qns.utils.log as log
from qns.utils.rnd import set_seed
from qns.network.protocol.proactive_routing import ProactiveRouting
from qns.network.protocol.link_layer import LinkLayer
from qns.network.protocol.proactive_routing_controller import ProactiveRoutingControllerApp
from qns.network.topology.customtopo import CustomTopology

from qns.entity.monitor import Monitor
from qns.entity.qchannel import RecvQubitPacket
from qns.network.protocol.event import QubitReleasedEvent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


log.logger.setLevel(logging.CRITICAL)

SEED_BASE = 100

light_speed = 2 * 10**5 # km/s

# parameters
sim_duration = 1
entg_attempt_rate = 50e6         # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)
init_fidelity = 0.99
t_coherence = 0.01    # sec
p_swap = 0.5


# 3-nodes topology
swapping_config = "isolation_1"
ch_1 = 32
ch_2 = 18

def generate_topology(channel_capacity = 1):
    return {
    "qnodes": [
        {
            "name": "S",
            "memory": {
                "decoherence_rate": 1 / t_coherence,
                # cutoff_ratio (float): the ratio between cutoff time and memory coherence time (default 1, should be between 0 and 1).
                # e.g., a qubit is considered effectively decohered when coherence drops below 1% or 0.1%, which happens at t ≈ 4.6T to 6.9T.
            },
            "apps": [LinkLayer(attempt_rate=entg_attempt_rate, init_fidelity=init_fidelity), ProactiveRouting()]
        },
        {
            "name": "R",
            "memory": {
                "decoherence_rate": 1 / t_coherence,
            },
            "apps": [LinkLayer(attempt_rate=entg_attempt_rate, init_fidelity=init_fidelity), ProactiveRouting(ps=p_swap)]
        },
        {
            "name": "D",
            "memory": {
                "decoherence_rate": 1 / t_coherence,
            },
            "apps": [LinkLayer(attempt_rate=entg_attempt_rate, init_fidelity=init_fidelity), ProactiveRouting()]
        }
    ],
    "qchannels": [
        { "node1": "S", "node2":"R", "capacity": channel_capacity, "parameters": {"length": ch_1, "delay": ch_1 / light_speed, "drop_rate": 1} },
        { "node1": "R", "node2":"D", "capacity": channel_capacity, "parameters": {"length": ch_2, "delay": ch_2 / light_speed, "drop_rate": 1} }
    ],
    "cchannels": [
        { "node1": "S", "node2":"R", "parameters": {"length": ch_1, "delay": ch_1 / light_speed} },
        { "node1": "R", "node2":"D", "parameters": {"length": ch_2, "delay": ch_2 / light_speed} },
        { "node1": "ctrl", "node2":"S", "parameters": {"length": 1.0, "delay": 1 / light_speed} },
        { "node1": "ctrl", "node2":"R", "parameters": {"length": 1.0, "delay": 1 / light_speed} },
        { "node1": "ctrl", "node2":"D", "parameters": {"length": 1.0, "delay": 1 / light_speed} }
    ],
    "controller": {
        "name": "ctrl",
        "apps": [ProactiveRoutingControllerApp(swapping=swapping_config)]
    }
    }

def run_simulation(num_qubits, seed):
    json_topology = generate_topology(channel_capacity = num_qubits)
    
    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)
    
    topo = CustomTopology(json_topology)
    net = QuantumNetwork(
        topo=topo,
        route=DijkstraRouteAlgorithm(),
        timing_mode=TimingModeEnum.ASYNC
    )
    net.install(s)

    # attempts rate per second per qchannel
    attempts_rate = {}    
    # etg rate per second per qchannel
    ent_rate = {}
    def watch_ent_rate(simulator, network, event):
        if event.qchannel.name in ent_rate:
            ent_rate[event.qchannel.name]+=1
            attempts_rate[event.qchannel.name]+=event.qubit.attempts
        else:
            ent_rate[event.qchannel.name] = 1
            attempts_rate[event.qchannel.name] = event.qubit.attempts

    m_ent_rate = Monitor(name="ent_rate", network=None)
    m_ent_rate.add_attribution(name="ent_rate", calculate_func=watch_ent_rate)
    m_ent_rate.at_event(RecvQubitPacket)
    m_ent_rate.install(s)

    s.run()

    attempts_rate.update({k: v / sim_duration for k, v in attempts_rate.items()})
    ent_rate.update({k: v / sim_duration for k, v in ent_rate.items()})

    # fraction of successful attempts per channel
    success_frac = {k: ent_rate[k] / attempts_rate[k] if attempts_rate[k] != 0 else 0 for k in ent_rate}
    return attempts_rate, ent_rate, success_frac


channel_map = {
    "q_S,R": 32,  # Channel name corresponding to 32 km link
    "q_R,D": 18   # Channel name corresponding to 18 km link
}

all_data = {
    "L": [],
    "M": [],
    "Attempts rate": [],
    "Entanglement rate": [],
    "Success rate": [],
    "Attempts std": [],
    "Ent std": [],
    "Success std": []
}

# Simulation loop
N_RUNS = 10
for M in range(1,6):
    stats = {
        32: {"attempts": [], "ent": [], "succ": []},
        18: {"attempts": [], "ent": [], "succ": []}
    }

    for i in range(N_RUNS):
        print(f"Sim: M={M}, run #{i+1}")
        seed = SEED_BASE + i
        attempts_rate, ent_rate, success_frac = run_simulation(M, seed)
        print(attempts_rate)
        for ch_name, L in channel_map.items():
            if ch_name in attempts_rate:
                stats[L]["attempts"].append(attempts_rate[ch_name])
                stats[L]["ent"].append(ent_rate[ch_name])
                stats[L]["succ"].append(success_frac[ch_name])
            else:
                print(f"Warning: channel {ch_name} not found in run_simulation output.")

    for L in [32, 18]:
        all_data["L"].append(L)
        all_data["M"].append(M)
        all_data["Attempts rate"].append(np.mean(stats[L]["attempts"]))
        all_data["Entanglement rate"].append(np.mean(stats[L]["ent"]))
        all_data["Success rate"].append(np.mean(stats[L]["succ"]))
        all_data["Attempts std"].append(np.std(stats[L]["attempts"]))
        all_data["Ent std"].append(np.std(stats[L]["ent"]))
        all_data["Success std"].append(np.std(stats[L]["succ"]))

# Convert to DataFrame
df = pd.DataFrame(all_data)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))

labels = {32: "L=32", 18: "L=18"}

for L in [32, 18]:
    df_L = df[df["L"] == L]
    axs[0].errorbar(df_L["M"], df_L["Attempts rate"], yerr=df_L["Attempts std"],
                    marker='o', linestyle='--', label=labels[L], capsize=3)
    axs[1].errorbar(df_L["M"], df_L["Entanglement rate"], yerr=df_L["Ent std"],
                    marker='o', linestyle='--', capsize=3)
    axs[2].errorbar(df_L["M"], df_L["Success rate"], yerr=df_L["Success std"],
                    marker='o', linestyle='--', capsize=3)

axs[0].set_title("Attempts rate")
axs[1].set_title("Ent. rate")
axs[2].set_title("Success rate")

for ax in axs:
    ax.set_xlabel("M")
axs[0].set_ylabel("Attempts/s")
axs[1].set_ylabel("Ent/s")
axs[2].set_ylabel("Fraction")
axs[2].legend()

fig.tight_layout()
plt.show()






# s.run_continuous()

# import signal
# def stop_emulation(sig, frame):
#     print('Stopping simulation...')
#     s.stop()
# signal.signal(signal.SIGINT, stop_emulation)

#results = []
#for req in net.requests:
#    src = req.src
#    results.append(src.apps[0].success_count)
#fair = sum(results)**2 / (len(results) * sum([r**2 for r in results]))
#log.monitor(requests_number, nodes_number, s.time_spend, sep=" ")
