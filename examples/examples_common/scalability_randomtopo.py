"""
Shared definitions for scalability_randomtopo experiments.
Both MQNS and SeQUeNCe simulation scripts can use these definitions.
"""

import time
from typing import Any, TypedDict

from tap import Tap

from mqns.entity.node import Controller
from mqns.network.network import QuantumNetwork
from mqns.network.proactive import (
    LinkLayer,
    MuxSchemeStatistical,
    ProactiveForwarder,
    ProactiveRoutingController,
)
from mqns.network.topology import ClassicTopology, RandomTopology
from mqns.utils import rng

# global simulation parameters
fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)
init_fidelity = 0.99
t_cohere = 5e-3
p_swap = 0.5
swapping_policy = "asap"
nqubits = 2000  # large enough to support qchannel capacity in random topology


# Command line arguments
class RunArgs(Tap):
    seed: int = -1  # random seed number
    nnodes: int = 16  # network size - number of nodes
    nedges: int = 20  # network size - number of edges
    sim_duration: float = 1.0  # simulation duration in seconds
    qchannel_capacity: int = 10  # quantum channel capacity
    time_limit: float = 10800.0  # wall-clock limit in seconds
    outdir: str = "."  # output directory

    @property
    def basename(self) -> str:
        """Output filename without extension."""
        return f"{self.qchannel_capacity}-{self.nnodes}-{self.nedges}-{self.seed}"


def parse_run_args() -> RunArgs:
    args = RunArgs().parse_args()
    if args.seed < 0:
        args.seed = int(time.time())
    return args


def build_network(args: RunArgs) -> QuantumNetwork:
    """
    Defines the topology with globally declared simulation parameters.
    """
    rng.reseed(args.seed)

    # Define topology.
    topo = RandomTopology(
        nodes_number=args.nnodes,
        lines_number=args.nedges,
        qchannel_args={"length": 30, "alpha": fiber_alpha},
        cchannel_args={"length": 30},
        memory_args={"capacity": nqubits, "t_cohere": t_cohere},
        nodes_apps=[
            LinkLayer(
                attempt_rate=entg_attempt_rate,
                init_fidelity=init_fidelity,
                eta_d=eta_d,
                eta_s=eta_s,
                frequency=frequency,
            ),
            ProactiveForwarder(ps=p_swap, mux=MuxSchemeStatistical()),
        ],
    )
    topo.controller = Controller("ctrl", apps=[ProactiveRoutingController()])

    # Construct network.
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow)
    topo.connect_controller(net.nodes)

    for qchannel in net.qchannels:
        qchannel.assign_memory_qubits(capacity=args.qchannel_capacity)

    # Compute routes using Dijkstra with hop count metric.
    net.build_route()

    # Generate random requests, proportional to network size.
    num_requests = max(2, int(args.nnodes / 10))
    net.random_requests(num_requests, min_hops=2, max_hops=5)

    return net


type RequestStats = tuple[float, float]
"""Per-request statistics: throughput, average fidelity."""


class RunResult(TypedDict):
    """Result from a simulation run."""

    time_spent: float
    """Total wall-clock time."""
    sim_progress: float
    """Finished timeline progress, 1.0 means all simulation finished."""
    requests: dict[str, RequestStats]
    """Per-request statistics."""
    nodes: dict[str, Any]
    """Per-node statistics."""
