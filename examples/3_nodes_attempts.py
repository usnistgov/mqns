import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

from qns.entity.monitor import Monitor
from qns.entity.qchannel import RecvQubitPacket
from qns.network.network import QuantumNetwork
from qns.simulator import Simulator
from qns.utils import log, set_seed

from examples_common.topo_3_nodes import build_topology


# Command line arguments
class Args(Tap):
    runs: int = 100  # number of trials per parameter set
    csv: str = ""  # save results as CSV file
    plt: str = ""  # save plot as image file


args = Args().parse_args()

log.set_default_level("DEBUG")

SEED_BASE = 100

# parameters
sim_duration = 1


def run_simulation(num_qubits: int, seed: int):
    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = build_topology(t_coherence=0.1, channel_qubits=num_qubits, swapping_order="no_swap")
    net = QuantumNetwork(topo=topo)
    net.install(s)

    # attempts rate per second per qchannel
    attempts_rate = {}
    # etg rate per second per qchannel
    ent_rate = {}

    def watch_ent_rate(simulator, network, event):
        if event.qchannel.name in ent_rate:
            ent_rate[event.qchannel.name] += 1
            attempts_rate[event.qchannel.name] += event.qubit.attempts
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
    "q_R,D": 18,  # Channel name corresponding to 18 km link
}

all_data = {
    "L": [],
    "M": [],
    "Attempts rate": [],
    "Entanglement rate": [],
    "Success rate": [],
    "Attempts std": [],
    "Ent std": [],
    "Success std": [],
}

# Simulation loop
for M in range(1, 6):
    stats = {32: {"attempts": [], "ent": [], "succ": []}, 18: {"attempts": [], "ent": [], "succ": []}}

    for i in range(args.runs):
        print(f"Sim: M={M}, run #{i + 1}")
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
if args.csv:
    df.to_csv(args.csv, index=False)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))

labels = {32: "L=32", 18: "L=18"}

for L in [32, 18]:
    df_L = df[df["L"] == L]
    axs[0].errorbar(
        df_L["M"], df_L["Attempts rate"], yerr=df_L["Attempts std"], marker="o", linestyle="--", label=labels[L], capsize=3
    )
    axs[1].errorbar(df_L["M"], df_L["Entanglement rate"], yerr=df_L["Ent std"], marker="o", linestyle="--", capsize=3)
    axs[2].errorbar(df_L["M"], df_L["Success rate"], yerr=df_L["Success std"], marker="o", linestyle="--", capsize=3)

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
if args.plt:
    fig.savefig(args.plt, dpi=300, transparent=True)
plt.show()


# s.run_continuous()

# import signal
# def stop_emulation(sig, frame):
#     print('Stopping simulation...')
#     s.stop()
# signal.signal(signal.SIGINT, stop_emulation)

# results = []
# for req in net.requests:
#    src = req.src
#    results.append(src.apps[0].success_count)
# fair = sum(results)**2 / (len(results) * sum([r**2 for r in results]))
# log.monitor(requests_number, nodes_number, s.time_spend, sep=" ")
