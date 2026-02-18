from tap import Tap

from mqns.network.builder import CTRL_DELAY, NetworkBuilder
from mqns.network.proactive import ProactiveForwarder
from mqns.network.protocol.link_layer import LinkLayerCounters
from mqns.network.route import YenRouteAlgorithm
from mqns.simulator import Simulator
from mqns.utils import log, rng

log.set_default_level("DEBUG")


class Args(Tap):
    sim_duration: float = 3  # simulation duration in seconds


SEED_BASE = 100

# Quantum channel lengths
ch_S_R1 = 10
ch_R1_R2 = 10
ch_R2_R3 = 10
ch_R3_R4 = 10
ch_R4_D = 10
ch_S_R5 = 15
ch_R5_R3 = 15


def run_simulation(seed: int, args: Args):
    rng.reseed(seed)

    net = (
        NetworkBuilder(
            route=YenRouteAlgorithm(k_paths=3),
        )
        .topo(
            mem_capacity=4,
            channels=[
                ("S-R1", ch_S_R1, 2),
                ("R1-R2", ch_R1_R2, 2),
                ("R2-R3", ch_R2_R3, (2, 1)),
                ("R3-R4", ch_R3_R4, 2),
                ("R4-D", ch_R4_D, (2, 4)),
                ("S-R5", ch_S_R5, 2),
                ("R5-R3", ch_R5_R3, (2, 1)),
            ],
            t_cohere=0.01,
        )
        .proactive_centralized()
        .path("S-D", swap="r2l")
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
