"""
Demonstrate ``MatrixTrafficGenerator`` usage.

1. Generate a sample traffic definition file::

    python mtg.py --tm +toml >mtg.tm.toml

2. Modify the traffic definition file.

   At least one traffic flow must have non-zero probability.

3. Run the simulation with traffic definition file::

    python mtg.py --tm mtg.tm.toml --csv mtg.csv --sim_duration 10 --rate 10

CSV output includes per-request statistics:

* Request attributes: end-nodes, active period, desired EPR quantity.

* Request state at end of simulation, which could be:

  * **REJECTED**: Controller rejected request due to insufficient resources.
  * **EPR_COUNT_REACHED**: Request is finished because desired EPR quantity is reached.
  * **EXPIRED**: Request is finished because active period has ended.
  * **ACTIVE**: Request is unfinished at end of simulation.

* Consumed EPR quantity, average fidelity.

* Latency since request arrival (start of active period) and first/last EPR delivery.

  * This includes ``CTRL_DELAY`` in Proactive-Centralized mode.
"""

import csv
import itertools
from typing import Literal, Self, cast, override

from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, PositiveInt, model_validator
from tap import Tap

from mqns.network.builder import NetworkBuilder, tap_configure
from mqns.network.network import MatrixTrafficGenerator, QuantumNetwork, Request, TrafficMatrixMapping
from mqns.network.proactive import MuxSchemeLiteral
from mqns.network.protocol.consumer import RequestCounters
from mqns.simulator import Simulator
from mqns.utils import log, rng, seed_env

from examples_common.param_def import load_param_def
from examples_common.topo_multiplexing import END_NODES, define_topo

log.set_default_level("CRITICAL")


class Args(Tap):
    tm: str = ""  # network traffic definition file
    sim_duration: float = 1.0  # simulation duration in seconds
    mode: Literal["PCA", "PCS", "RCS"] = "PCA"
    sync_timing: list[float]
    M: tuple[int, int] = (50, 32)  # channel capacity
    mux: MuxSchemeLiteral = "B"  # multiplexing scheme
    rate: float = 1.0  # request arrival rate (req/s)
    csv: str = ""  # save results as CSV file

    @override
    def configure(self) -> None:
        tap_configure(self)


class NetTrafficDef(BaseModel):
    matrix: dict[str, NonNegativeFloat]
    """
    Traffic matrix.
    Key: source and destination node, delimited by hyphen.
    Value: relative weight.
    """
    duration: PositiveFloat
    """
    Active duration per request, in seconds.
    """
    epr_count: Literal[-1] | PositiveInt
    """
    Desired quantity of entangled pairs per request.
    ``-1`` means infinite.
    """

    @model_validator(mode="after")
    def _(self) -> Self:
        if sum(self.matrix.values()) <= 0:
            raise ValueError("matrix must contain at least one non-zero probability")
        return self


def make_tm_sample() -> NetTrafficDef:
    matrix: dict[str, float] = {}
    prob = 1.0
    for src, dst in itertools.product(END_NODES, END_NODES):
        if src == dst:
            continue
        matrix[f"{src}-{dst}"] = prob
        prob = 0.0
    return NetTrafficDef(matrix=matrix, duration=1, epr_count=-1)


def build_network(args: Args) -> QuantumNetwork:
    b = NetworkBuilder()
    define_topo(b, ch_capacity=args.M)
    match args.mode:
        case "PCA":
            b.proactive_centralized(mux=args.mux)
        case "PCS":
            b.proactive_centralized(mux=args.mux, timing=args.sync_timing)
        case "RCS":
            # XXX desired EPR count is unenforced
            b.reactive_centralized(timing=args.sync_timing)
    return b.make_network()


type Result = list[tuple[Request, RequestCounters]]


def run_simulation(seed: int, args: Args, tm: NetTrafficDef) -> Result:
    rng.reseed(seed)
    net = build_network(args)

    mtg = MatrixTrafficGenerator(
        net,
        cast(TrafficMatrixMapping, tm.matrix),
        sched="eager",
        rp_args={"bufferspace_mv": 1},
        rate=args.rate,
        duration=tm.duration,
        epr_count=tm.epr_count,
    )

    s = Simulator(0, args.sim_duration, install_to=(log, net, mtg))
    s.run()

    return [(req, RequestCounters.of(net, req)) for req in net.requests]


def save_csv(args: Args, result: Result) -> None:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            (
                "req_id",
                "src",
                "dst",
                "active_since",
                "active_until",
                "req_epr_count",
                "state",
                "n_consumed",
                "fid",
                "lat_first",
                "lat_last",
            )
        )
        for req, cnt in result:
            lat_first, lat_last = cnt.get_latency(req.active_since)
            w.writerow(
                (
                    req.req_id,
                    req.src,
                    req.dst,
                    req.active_since,
                    req.active_until,
                    req.epr_count,
                    req.state.name,
                    cnt.n_consumed,
                    cnt.consumed_avg_fidelity,
                    lat_first,
                    lat_last,
                )
            )


def main(args: Args) -> None:
    tm = load_param_def(args.tm, make_tm_sample())
    if tm is None:
        return None

    result = run_simulation(seed_env(0), args, tm)
    if args.csv:
        save_csv(args, result)

    total_consumed = sum(cnt.n_consumed for _, cnt in result)
    print(f"Simulation completed, {len(result)} requests, {total_consumed} consumed")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
