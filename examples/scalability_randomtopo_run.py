import json
import os.path
import time

from tap import Tap

from mqns.entity.node import Controller
from mqns.network.network import QuantumNetwork
from mqns.network.proactive import (
    LinkLayer,
    MuxSchemeStatistical,
    ProactiveForwarder,
    ProactiveRoutingController,
    QubitAllocationType,
    RoutingPathSingle,
)
from mqns.network.topology import ClassicTopology, RandomTopology
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

"""
This script is typically invoked as part of scalability_randomtopo experiment.
See scalability_randomtopo.sh for how to run this script.

If --profiling flag is specified, this script instead performs deterministic profiling.
It prints the profiling statistics to the console, and does not generate any output file.
"""

log.set_default_level("CRITICAL")


# Command line arguments
class Args(Tap):
    seed: int = -1  # random seed number
    nnodes: int = 16  # network size - number of nodes
    nedges: int = 20  # network size - number of edges
    sim_duration: float = 1.0  # simulation duration in seconds
    qchannel_capacity: int = 10  # quantum channel capacity
    outdir: str = "."  # output directory


args = Args().parse_args()
if args.seed < 0:
    args.seed = int(time.time())

# parameters
fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)

init_fidelity = 0.99
t_cohere = 5e-3  # 10e-3

p_swap = 0.5
swapping_policy = "asap"

nqubits = 2000  # large enough to support qchannel capacity in random topology


def build_network() -> QuantumNetwork:
    """
    Defines the topology with globally declared simulation parameters.
    """

    topo = RandomTopology(
        nodes_number=args.nnodes,
        lines_number=args.nedges,
        qchannel_args={"length": 30},
        cchannel_args={"length": 30},
        memory_args={"capacity": nqubits, "t_cohere": t_cohere},
        nodes_apps=[
            LinkLayer(
                attempt_rate=entg_attempt_rate,
                init_fidelity=init_fidelity,
                alpha_db_per_km=fiber_alpha,
                eta_d=eta_d,
                eta_s=eta_s,
                frequency=frequency,
            ),
            ProactiveForwarder(ps=p_swap, mux=MuxSchemeStatistical()),
        ],
    )
    topo.controller = Controller("ctrl", apps=[ProactiveRoutingController()])

    # Default: Dijkstra with hop count metric
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)
    topo.connect_controller(net.nodes)

    for qchannel in net.qchannels:
        qchannel.assign_memory_qubits(capacity=args.qchannel_capacity)

    return net


def run_simulation() -> dict:
    # Assign random seed.
    set_seed(args.seed)
    s = Simulator(0, args.sim_duration + 5e-06, accuracy=1000000)
    log.install(s)

    # Generate random topology.
    net = build_network()
    net.install(s)

    # Generate random requests, proportional to network size.
    num_requests = max(2, int(args.nnodes / 10))
    net.random_requests(num_requests, min_hops=2, max_hops=5)

    # Install paths for requests.
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    for req in net.requests:
        ctrl.install_path(
            RoutingPathSingle(req.src.name, req.dst.name, qubit_allocation=QubitAllocationType.DISABLED, swap="asap")
        )

    # Run the simulation.
    s.run()

    # Collect wall-clock duration and per-request statistics.
    stats = dict[str, tuple[float, float]]()
    for req in net.requests:
        fw = req.src.get_app(ProactiveForwarder)
        stats[f"{req.src.name}-{req.dst.name}"] = fw.cnt.n_consumed / args.sim_duration, fw.cnt.consumed_avg_fidelity
    return {
        "time_spent": s.time_spend,
        "requests": stats,
    }


if __name__ == "__main__":
    basename = f"{args.qchannel_capacity}-{args.nnodes}-{args.nedges}-{args.seed}"
    result = run_simulation()
    with open(os.path.join(args.outdir, f"{basename}.json"), "w") as file:
        json.dump(result, file)
