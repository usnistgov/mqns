import configparser
import itertools
import json
import os.path
import sys
import time
from typing import cast

from sequence.network_management.network_manager import ResourceReservationProtocol
from sequence.network_management.reservation import Reservation
from sequence.topology.node import QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo

from mqns.network.network import QuantumNetwork, Request
from mqns.utils import WallClockTimeout

from sequence_detail.resource_reservation import create_rules
from sequence_detail.scalability_randomtopo import (
    EntanglementRequestApp,
    ResetApp,
    set_parameters,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples_common.scalability_randomtopo import (
    RequestStats,
    RunArgs,
    RunResult,
    parse_run_args,
)
from examples_common.scalability_randomtopo import build_network as mqns_build_network

"""
This script is part of scalability_randomtopo experiment for comparison with SeQUeNCe simulator.
It can be invoked in the same way as mqns/examples/scalability_randomtopo_run.py .
The topology and end-to-end entanglement requests are generated in MQNS and then converted to SeQUeNCe.
"""


COUNTER_NAMES = (
    "success_number",
    "attempts_number",
    "failed_attempts",
    "emit_number",
    "success_swapping",
    "expired_memories_counter",
)

common_config = configparser.ConfigParser()
common_config.read_string("""
[Memory]
efficiency = 0.95
fidelity = 0.99
frequency = 1e6
wavelength = 1550
coherence_time = 0.005

[Detector]
efficiency = 0.95
count_rate = 60e6
resolution = 100

[Swapping]
success_rate = 0.5

[qchannel]
attenuation = 0.2
frequency = 50e6
""")


class TimelineBounds:
    """Simulation time boundary."""

    def __init__(self, sim_duration: float):
        self.start_t = int(0.1e12)
        """Simulation start time in picoseconds."""
        self.stop_t = self.start_t + int(sim_duration * 1e12)
        """Simulation stop time in picoseconds."""


def convert_network(net: QuantumNetwork, tlb: TimelineBounds) -> dict:
    """
    Convert MQNS network topology into SeQUeNCe network topology.
    """

    nodes = [
        {
            "name": node.name,
            "type": "QuantumRouter",
            "seed": 0,
            "memo_size": 2000,
        }
        for node in net.nodes
    ]

    qconnections = [
        {
            "node1": ch.node_list[0].name,
            "node2": ch.node_list[1].name,
            "distance": 30000,
            "attenuation": 0.0002,
            "type": "meet_in_the_middle",
        }
        for ch in net.qchannels
    ]

    cchannels: list[dict] = []
    for src, dst in itertools.product(net.nodes, net.nodes):
        if src == dst:
            continue
        metric, _, _ = net.query_route(src, dst)[0]
        cchannels.append(
            {
                "source": src.name,
                "destination": dst.name,
                "distance": 30000 * int(metric),
            }
        )

    return {
        "nodes": nodes,
        "qconnections": qconnections,
        "cchannels": cchannels,
        "is_parallel": False,
        "stop_time": tlb.stop_t,
    }


def build_network(basename: str, net: QuantumNetwork, tlb: TimelineBounds) -> tuple[RouterNetTopo, list[QuantumRouter]]:
    """
    Build SeQUeNCe network topology that matches MQNS network topology.
    """

    filename = os.path.join(args.outdir, f"{basename}.sequence-topo.json")
    network_json = convert_network(net, tlb)
    with open(filename, "w") as f:
        json.dump(network_json, f)

    topo = RouterNetTopo(filename)
    set_parameters(topo, common_config)
    ResourceReservationProtocol.create_rules = create_rules
    setattr(Reservation, "link_capacity", args.qchannel_capacity)
    setattr(Reservation, "swapping_order", "ASAP")

    routers = cast(list[QuantumRouter], topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER))
    for router in routers:
        setattr(router.network_manager, "network_routers", routers)
        for CNT in COUNTER_NAMES:
            setattr(router, CNT, 0)

    return topo, routers


type RequestApps = tuple[EntanglementRequestApp, ResetApp]


def convert_request(routers: list[QuantumRouter], request: Request) -> RequestApps:
    """
    Convert MQNS src-dst request into a pair of applications in SeQUeNCe.
    """
    src_node = next((r for r in routers if r.name == request.src.name))
    dst_node = next((r for r in routers if r.name == request.dst.name))
    return (
        EntanglementRequestApp(src_node, dst_node.name),
        ResetApp(dst_node, src_node.name),
    )


def start_requests(requests: list[RequestApps], tlb: TimelineBounds) -> None:
    """
    Start a pair of applications for src-dst request.
    """
    for src, dst in requests:
        src.start(dst.node.name, tlb.start_t, tlb.stop_t, memo_size=1, fidelity=0.1)


def run_simulation(args: RunArgs) -> RunResult:
    # Generate random topology and requests in MQNS.
    # MQNS random seed is set within.
    net = mqns_build_network(args)

    # Build the same topology and requests in SeQUeNCe.
    tlb = TimelineBounds(args.sim_duration)
    topo, routers = build_network(args.basename, net, tlb)
    requests = [convert_request(routers, req) for req in net.requests]
    del net

    # Initialize timeline and start request applications.
    tl = topo.get_timeline()
    tl.stop_time = tlb.stop_t
    tl.init()
    start_requests(requests, tlb)

    # Run the simulation timeline.
    timeout = WallClockTimeout(args.time_limit, stop=tl.stop)
    trs = time.time()
    with timeout():
        tl.run()
    tre = time.time()
    sim_progress = (tl.time - tlb.start_t) / (tlb.stop_t - tlb.start_t) if timeout.occurred else 1.0
    sim_duration = args.sim_duration * sim_progress

    # Collect results.
    def gather_request_stats(src: EntanglementRequestApp) -> RequestStats:
        return src.memory_counter / sim_duration, src.get_fidelity()

    return RunResult(
        time_spent=tre - trs,
        sim_progress=sim_progress,
        requests={f"{src.node.name}-{dst.node.name}": gather_request_stats(src) for src, dst in requests},
        nodes={router.name: {CNT: getattr(router, CNT) for CNT in COUNTER_NAMES} for router in routers},
    )


if __name__ == "__main__":
    args = parse_run_args()
    result = run_simulation(args)
    with open(os.path.join(args.outdir, f"{args.basename}.sequence.json"), "w") as file:
        json.dump(result, file)
