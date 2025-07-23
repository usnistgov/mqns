import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

from qns.network.network import QuantumNetwork
from qns.network.protocol import ProactiveForwarder
from qns.simulator import Simulator
from qns.utils import log, set_seed

from examples_common.stats import gather_etg_decoh
from examples_common.topo_3_nodes import build_topology


# Command line arguments
class Args(Tap):
    runs: int = 10  # number of trials per parameter set
    csv: str = ""  # save results as CSV file
    plt: str = ""  # save plot as image file


args = Args().parse_args()

log.set_default_level("DEBUG")

SEED_BASE = 100

# parameters
sim_duration = 3


def run_simulation(t_coherence: float, seed: int):
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
    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = build_topology(t_coherence=t_coherence, channel_qubits=1)
    net = QuantumNetwork(topo=topo)
    net.install(s)

    s.run()

    #### get stats
    _, _, decoh_ratio = gather_etg_decoh(net)
    e2e_rate = net.get_node("S").get_app(ProactiveForwarder).cnt.n_consumed / sim_duration
    return e2e_rate, decoh_ratio


########################### Main #########################
results = {"T_cohere": [], "Mean Rate": [], "Std Rate": []}

# t_cohere_values = [2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 8e-2, 1e-1]
t_cohere_values = np.geomspace(2e-3, 1e-1, 8)

for t_cohere in t_cohere_values:
    rates = []
    for i in range(args.runs):
        print(f"T_cohere={t_cohere:.4f}, run {i + 1}")
        seed = SEED_BASE + i
        rate, *_ = run_simulation(t_cohere, seed)
        rates.append(rate)

    results["T_cohere"].append(t_cohere)
    results["Mean Rate"].append(np.mean(rates))
    results["Std Rate"].append(np.std(rates))

# Convert to DataFrame
df = pd.DataFrame(results)
if args.csv:
    df.to_csv(args.csv, index=False)

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
if args.plt:
    plt.savefig(args.plt, dpi=300, transparent=True)
plt.show()
