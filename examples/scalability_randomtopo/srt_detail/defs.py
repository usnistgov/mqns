"""
Shared definitions for scalability_randomtopo experiments.
Both MQNS and SeQUeNCe simulation scripts can use these definitions.
"""

import time
import tomllib
from typing import Any, NotRequired, TypedDict, cast, override

from tap import Tap

from mqns.network.builder import NetworkBuilder, RandomTopoGen
from mqns.network.network import QuantumNetwork, generate_random_requests
from mqns.utils import rng


class NetworkSize(TypedDict):
    nodes: int
    edges: int


class Params(TypedDict):
    seed_base: int
    runs: int
    sim_duration: float
    qchannel_capacity: int
    time_limit: float
    enable_sequence: bool
    network_sizes: list[NetworkSize]
    cpuset_cpus: NotRequired[list[int]]


def load_params(filename: str) -> Params:
    with open(filename, "rb") as f:
        return cast(Params, tomllib.load(f))


class ParamsArgs(Tap):
    params: Params

    @property
    def sim_duration(self) -> float:
        return self.params["sim_duration"]

    @property
    def qchannel_capacity(self) -> int:
        return self.params["qchannel_capacity"]

    @property
    def time_limit(self) -> float:
        return self.params["time_limit"]

    @override
    def configure(self) -> None:
        self.add_argument("--params", type=load_params)


class RunArgs(ParamsArgs):
    seed: int = -1  # random seed number
    nodes: int = 16  # network size - number of nodes
    edges: int = 20  # network size - number of edges
    outdir: str = "."  # output directory

    @property
    def basename(self) -> str:
        """Output filename without extension."""
        return f"{self.qchannel_capacity}-{self.nodes}-{self.edges}-{self.seed}"

    @override
    def process_args(self) -> None:
        if self.seed < 0:
            self.seed = int(time.time())


# global simulation parameters
# note: only applied to MQNS, not auto-converted to SeQUeNCe
fiber_alpha = 0.2
eta_s = 0.95
eta_d = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz) -- ineffective
init_fidelity = 0.99
t_cohere = 5e-3
p_swap = 0.5
swapping_policy = "asap"
nqubits = 2000  # large enough to support qchannel capacity in random topology


def build_network(args: RunArgs) -> QuantumNetwork:
    """
    Defines the topology with globally declared simulation parameters.
    """
    rng.reseed(args.seed)

    # Construct network.
    net = (
        NetworkBuilder()
        .topo_from(
            RandomTopoGen(args.nodes, args.edges),
            mem_capacity=nqubits,
            t_cohere=t_cohere,
            frequency=frequency,
            ch_length=30,
            ch_capacity=args.qchannel_capacity,
            fiber_alpha=fiber_alpha,
            eta_s=eta_s,
            eta_d=eta_d,
            init_fidelity=init_fidelity,
        )
        .proactive_centralized(mux="S", p_swap=p_swap)
        .make_network()
    )

    # Generate random requests, proportional to network size.
    num_requests = max(2, int(args.nodes / 10))
    net.add_request(*generate_random_requests(net, num_requests, metric_min=2, metric_max=5))

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
