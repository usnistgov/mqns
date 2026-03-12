import math
from typing import Literal, override

from tap import Tap

from mqns.network.builder import EprTypeLiteral, NetworkBuilder, tap_configure
from mqns.network.fw import Forwarder, ForwarderCounters
from mqns.network.protocol.classicbridge import ClassicBridge
from mqns.simulator import Simulator
from mqns.utils import log, rng

log.set_default_level("INFO")


class Args(Tap):
    nats_prefix: str = ClassicBridge.DEFAULT_NATS_PREFIX  # prefix of NATS subjects
    sim_accuracy: int = 1_000_000  # simulation accuracy in time slots per second
    seed: int | None = None  # random seed
    mode: Literal["P", "R"] = "P"  # choose proactive or reactive mode
    epr_type: EprTypeLiteral  # network-wide EPR type

    @override
    def configure(self) -> None:
        super().configure()
        tap_configure(self)


def run_simulation(args: Args) -> dict[str, ForwarderCounters]:
    rng.reseed(args.seed)

    b = NetworkBuilder(
        epr_type=args.epr_type,
    )
    b.topo(
        channels=[
            ("S1-R1", 50, 2),
            ("S2-R1", 50, 2),
            ("R1-R2", 10, 4),
            ("R2-D1", 50, 2),
            ("R2-D2", 50, 2),
        ]
    )

    match args.mode:
        case "P":
            b.proactive_centralized()
        case "R":
            b.reactive_centralized()

    b.external_controller(nats_prefix=args.nats_prefix)

    net = b.make_network()
    del b

    s = Simulator(0, math.inf, accuracy=args.sim_accuracy, install_to=(log, net))
    s.run()

    results: dict[str, ForwarderCounters] = {}
    for node in net.nodes:
        results[node.name] = node.get_app(Forwarder).cnt
    return results


if __name__ == "__main__":
    args = Args().parse_args()
    results = run_simulation(args)
    print("")
    print("---- RESULTS ----")
    for node_name in ("S1", "D1", "S2", "D2", "R1", "R2"):
        print(f"[{node_name}] {results[node_name]}")
