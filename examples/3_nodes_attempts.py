from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

from mqns.entity.monitor import Monitor
from mqns.network.network import QuantumNetwork
from mqns.network.protocol.event import LinkArchSuccessEvent
from mqns.simulator import Event, Simulator
from mqns.utils import log, set_seed

from examples_common.topo_linear import build_topology


# Command line arguments
class Args(Tap):
    runs: int = 50  # number of trials per parameter set
    csv: str = ""  # save results as CSV file
    plt: str = ""  # save plot as image file


args = Args().parse_args()

log.set_default_level("CRITICAL")

SEED_BASE = 100

# parameters
sim_duration = 1


def run_simulation(num_qubits: int, seed: int):
    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = build_topology(
        nodes=["S", "R", "D"],
        t_coherence=0.1,
        channel_length=[32, 18],
        channel_capacity=num_qubits,
        swap="no_swap",
    )
    net = QuantumNetwork(topo=topo)
    net.install(s)

    counts = defaultdict[str, list[int]](lambda: [0, 0])

    def watch_ent_rate(simulator: Simulator, network: QuantumNetwork | None, event: Event):
        _ = simulator
        _ = network
        assert isinstance(event, LinkArchSuccessEvent)
        if event.node != event.epr.dst:  # only count at dst-node
            return
        assert event.epr.attempts is not None
        record = counts[event.node.name]
        record[0] += 1
        record[1] += event.epr.attempts

    m_ent_rate = Monitor(name="ent_rate", network=None)
    m_ent_rate.add_attribution(name="ent_rate", calculate_func=watch_ent_rate)
    m_ent_rate.at_event(LinkArchSuccessEvent)
    m_ent_rate.install(s)

    s.run()

    attempts_rate = {k: v[1] / sim_duration for k, v in counts.items()}
    ent_rate = {k: v[0] / sim_duration for k, v in counts.items()}

    # fraction of successful attempts per dst-node
    success_frac = {k: ent_rate[k] / attempts_rate[k] if attempts_rate[k] != 0 else 0 for k in ent_rate}
    return attempts_rate, ent_rate, success_frac


node_map = {
    "R": 32,  # dst-node corresponding to 32 km link
    "D": 18,  # dst-node corresponding to 18 km link
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


########### Simulation loop
results_summary = []  # to store formatted result strings for final print

for M in range(1, 6):
    stats = {32: {"attempts": [], "ent": [], "succ": []}, 18: {"attempts": [], "ent": [], "succ": []}}

    for i in range(args.runs):
        print(f"Sim: M={M}, run #{i + 1}")
        seed = SEED_BASE + i
        attempts_rate, ent_rate, success_frac = run_simulation(M, seed)
        for node_name, L in node_map.items():
            if node_name in attempts_rate:
                stats[L]["attempts"].append(attempts_rate[node_name])
                stats[L]["ent"].append(ent_rate[node_name])
                stats[L]["succ"].append(success_frac[node_name])
            else:
                print(f"Warning: dst-node {node_name} not found in run_simulation output.")

    for L in [32, 18]:
        att_mean, att_std = np.mean(stats[L]["attempts"]), np.std(stats[L]["attempts"])
        ent_mean, ent_std = np.mean(stats[L]["ent"]), np.std(stats[L]["ent"])
        succ_mean, succ_std = np.mean(stats[L]["succ"]), np.std(stats[L]["succ"])

        all_data["L"].append(L)
        all_data["M"].append(M)
        all_data["Attempts rate"].append(att_mean)
        all_data["Entanglement rate"].append(ent_mean)
        all_data["Success rate"].append(succ_mean)
        all_data["Attempts std"].append(att_std)
        all_data["Ent std"].append(ent_std)
        all_data["Success std"].append(succ_std)

        results_summary.append(
            f"L={L:<2}  M={M:<2}   C: {att_mean:.1f} ({att_std:.1f})   "
            f"P: {succ_mean:.4f} ({succ_std:.4f})   E: {ent_mean:.1f} ({ent_std:.1f})"
        )


# Convert to DataFrame
df = pd.DataFrame(all_data)
if args.csv:
    df.to_csv(args.csv, index=False)

# Final results summary print
print("\n=== Simulation Summary ===")
for line in results_summary:
    print(line)


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
