import numpy as np
import pandas as pd
from tap import Tap

from qns.network.network import QuantumNetwork
from qns.network.protocol import LinkLayer, ProactiveForwarder
from qns.simulator import Simulator
from qns.utils import log, set_seed

from examples_common.topo_3_nodes import build_topology


# Command line arguments
class Args(Tap):
    runs: int = 1  # number of trials per parameter set
    csv: str = ""  # save results as CSV file


args = Args().parse_args()

log.set_default_level("DEBUG")

SEED_BASE = 100

# parameters
sim_duration = 5


def run_simulation(t_coherence: float, seed: int):
    set_seed(seed)
    s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    topo = build_topology(
        t_coherence=t_coherence,
        channel_qubits=2,
        init_fidelity=0.7,
    )
    net = QuantumNetwork(topo=topo)
    net.install(s)

    s.run()

    #### get stats
    total_etg = 0
    total_decohered = 0
    for node in net.get_nodes():
        ll_app = node.get_app(LinkLayer)
        total_etg += ll_app.etg_count
        total_decohered += ll_app.decoh_count

    fw_s = net.get_node("S").get_app(ProactiveForwarder)
    e2e_rate = fw_s.cnt.n_consumed / sim_duration
    mean_fidelity = fw_s.cnt.consumed_avg_fidelity

    return e2e_rate, mean_fidelity, total_decohered / total_etg if total_etg > 0 else 0


results = {"T_cohere": [], "Mean Rate": [], "Std Rate": [], "Mean F": [], "Std F": []}

t_cohere_values = [1]
# t_cohere_values = [2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 8e-2, 1e-1]
# t_cohere_values = np.geomspace(2e-3, 1e-1, 8)

for t_cohere in t_cohere_values:
    rates = []
    fids = []
    for i in range(args.runs):
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
if args.csv:
    df.to_csv(args.csv, index=False)

print(df)
