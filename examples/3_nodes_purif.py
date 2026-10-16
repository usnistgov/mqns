import numpy as np
import pandas as pd
from tap import Tap

from mqns.network.builder import CTRL_DELAY, NetworkBuilder
from mqns.network.protocol.consumer import RequestCounters
from mqns.simulator import Simulator
from mqns.utils import log, rng, seed_seq_env


# Command line arguments
class Args(Tap):
    runs: int = 1  # number of trials per parameter set
    csv: str = ""  # save results as CSV file


args = Args().parse_args()

log.set_default_level("DEBUG")

# parameters
sim_duration = 5


def run_simulation(seed: int, t_cohere: float):
    rng.reseed(seed)

    net = (
        NetworkBuilder()
        .topo_linear(
            nodes=3,
            t_cohere=t_cohere,
            channels=[32, 18],
            ch_capacity=2,
            init_fidelity=0.7,
        )
        .proactive_centralized()
        .request("S-D")
        .make_network()
    )

    s = Simulator(0, sim_duration + CTRL_DELAY, accuracy=1000000, install_to=(log, net))
    s.run()

    #### get stats
    req_cnt = RequestCounters.of(net, 0, "S-D")
    return req_cnt.get_rate(sim_duration), req_cnt.consumed_avg_fidelity


results = {"T_cohere": [], "Mean Rate": [], "Std Rate": [], "Mean F": [], "Std F": []}

t_cohere_values = [1]
# t_cohere_values = [2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 8e-2, 1e-1]
# t_cohere_values = np.geomspace(2e-3, 1e-1, 8)

for t_cohere in t_cohere_values:
    rates = []
    fids = []
    for seed in seed_seq_env(args.runs, 100):
        print(f"T_cohere={t_cohere:.4f}, seed={seed}")
        rate, f = run_simulation(seed, t_cohere)
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
