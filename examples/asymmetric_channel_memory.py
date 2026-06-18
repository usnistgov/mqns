"""
This script simulates a 4-node linear network with highly asymmetric links and memory capacities.

.. figure:: /_static/examples/asymmetric_channel_memory.png
   :alt: 4-node asymmetric network topology
   :align: center
   :width: 100%

The network model spans three quantum channels with distinct configurations:

* Fiber links of varying lengths: 32 km, 18 km, and 10 km.
* Heterogeneous, directional channel capacities restricting local qubit allocations.

The simulation executes under a proactive centralized configuration with a rigid left-to-right
(`l2r`) swapping strategy. It runs for a single trail with a memory coherence time fixed
at 10 ms, evaluating how localized structural bottlenecks and asymmetric resource limits
impact the end-to-end entanglement rate and qubit decoherence ratios.
"""

from tap import Tap

from mqns.network.builder import CTRL_DELAY, NetworkBuilder
from mqns.network.fw import ForwarderConsumeCounters
from mqns.network.protocol.link_layer import LinkLayerCounters
from mqns.simulator import Simulator
from mqns.utils import log, rng

log.set_default_level("DEBUG")


class Args(Tap):
    sim_duration: float = 3  # simulation duration in seconds


SEED_BASE = 100


def run_simulation(seed: int, args: Args):
    rng.reseed(seed)

    net = (
        NetworkBuilder()
        .topo_linear(
            nodes=4,
            channels=[(32, (4, 3)), (18, (1, 2)), (10, (2, 4))],
            t_cohere=0.01,
        )
        .proactive_centralized()
        .request("S-D", swap="l2r")
        .make_network()
    )

    s = Simulator(0, args.sim_duration + CTRL_DELAY, accuracy=1000000, install_to=(log, net))
    s.run()

    #### get stats
    consume_cnt = ForwarderConsumeCounters.of_path(net, "S", "D")
    ll_cnt = LinkLayerCounters.aggregate(net.nodes)
    return consume_cnt.get_rate(args.sim_duration), ll_cnt.decoh_ratio


if __name__ == "__main__":
    args = Args().parse_args()

    e2e_rate, decoh_ratio = run_simulation(SEED_BASE, args)
    print(f"E2E etg rate: {e2e_rate}")
    print(f"Expired memories: {decoh_ratio}")
