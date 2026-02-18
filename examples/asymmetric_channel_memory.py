from tap import Tap

from mqns.network.builder import CTRL_DELAY, NetworkBuilder
from mqns.network.proactive import ProactiveForwarder
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
            channel_length=[32, 18, 10],
            channel_capacity=[(4, 3), (1, 2), (2, 4)],
            t_cohere=0.01,
        )
        .proactive_centralized()
        .path("S-D", swap="swap_2_l2r")
        .make_network()
    )

    s = Simulator(0, args.sim_duration + CTRL_DELAY, accuracy=1000000, install_to=(log, net))
    s.run()

    #### get stats
    decoh_ratio = LinkLayerCounters.aggregate(net.nodes).decoh_ratio
    e2e_rate = net.get_node("S").get_app(ProactiveForwarder).cnt.n_consumed / args.sim_duration
    return e2e_rate, decoh_ratio


if __name__ == "__main__":
    args = Args().parse_args()

    e2e_rate, decoh_ratio = run_simulation(SEED_BASE, args)
    print(f"E2E etg rate: {e2e_rate}")
    print(f"Expired memories: {decoh_ratio}")
