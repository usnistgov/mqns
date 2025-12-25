import configparser
import itertools
import json
import os.path
import threading
import time
from typing import cast

from sequence.network_management.network_manager import ResourceReservationProtocol
from sequence.network_management.reservation import Reservation
from sequence.topology.node import QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo
from tap import Tap

from mqns.network.network import QuantumNetwork, Request
from mqns.network.topology import ClassicTopology, RandomTopology
from mqns.utils import set_seed

from sequence_detail.resource_reservation import create_rules
from sequence_detail.scalability_randomtopo import (
    EntanglementRequestApp,
    ResetApp,
    set_parameters,
)

"""
This script is part of scalability_randomtopo experiment for comparison with SeQUeNCe simulator.
It can be invoked in the same way as mqns/examples/scalability_randomtopo_run.py .
The topology and end-to-end entanglement requests are generated MQNS, and then converted to SeQUeNCe.
"""


# Command line arguments
class Args(Tap):
    seed: int = -1  # random seed number
    nnodes: int = 16  # network size - number of nodes
    nedges: int = 20  # network size - number of edges
    sim_duration: float = 1.0  # simulation duration in seconds
    qchannel_capacity: int = 10  # quantum channel capacity
    time_limit: float = 10800.0  # wall-clock limit in seconds
    outdir: str = "."  # output directory


args = Args().parse_args()
if args.seed < 0:
    args.seed = int(time.time())

start_t = int(0.1e12)
"""Simulation start time in picoseconds."""
stop_t = start_t + int(args.sim_duration * 1e12)
"""Simulation stop time in picoseconds."""

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


def convert_network(net: QuantumNetwork) -> dict:
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
        "stop_time": stop_t,
    }


def build_network(basename: str, net: QuantumNetwork) -> tuple[RouterNetTopo, list[QuantumRouter]]:
    """
    Build SeQUeNCe network topology that matches MQNS network topology.
    """

    filename = os.path.join(args.outdir, f"{basename}.topo.json")
    network_json = convert_network(net)
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


def convert_request(routers: list[QuantumRouter], request: Request) -> tuple[EntanglementRequestApp, ResetApp]:
    """
    Convert MQNS src-dst request into a pair of applications in SeQUeNCe.
    """
    src_node = next((r for r in routers if r.name == request.src.name))
    dst_node = next((r for r in routers if r.name == request.dst.name))
    return (
        EntanglementRequestApp(src_node, dst_node.name),
        ResetApp(dst_node, src_node.name),
    )


def start_requests(requests: list[tuple[EntanglementRequestApp, ResetApp]]) -> None:
    """
    Start a pair of applications for src-dst request.
    """
    for app_src, app_dst in requests:
        app_src.start(app_dst.node.name, start_t, stop_t, memo_size=1, fidelity=0.1)
        app_dst.set(start_t, stop_t)


def run_simulation(basename: str) -> dict:
    # Assign random seed.
    set_seed(args.seed)

    # Generate random topology.
    net = QuantumNetwork(
        topo=RandomTopology(
            nodes_number=args.nnodes,
            lines_number=args.nedges,
        ),
        classic_topo=ClassicTopology.Follow,
    )
    net.build_route()

    # Generate random requests, proportional to network size.
    num_requests = max(2, int(args.nnodes / 10))
    net.random_requests(num_requests, min_hops=2, max_hops=5)

    # Build the same topology and requests in SeQUeNCe.
    topo, routers = build_network(basename, net)
    requests = [convert_request(routers, req) for req in net.requests]

    # Initialize timeline and initialize request applications.
    tl = topo.get_timeline()
    tl.stop_time = stop_t
    tl.init()
    start_requests(requests)

    # Enforce maximum wall-clock time limit.
    timeout_occurred = [False]
    timeout_cancel = threading.Event()

    def stop_timeline_after_timeout():
        if not timeout_cancel.wait(timeout=args.time_limit):
            tl.stop()
            timeout_occurred[0] = True

    timeout_thread = threading.Thread(target=stop_timeline_after_timeout, daemon=True)

    # Run the simulation timeline.
    trs = time.time()
    timeout_thread.start()
    tl.run()
    timeout_cancel.set()
    tre = time.time()

    # Collect wall-clock duration and per-router counters.
    d: dict = {
        "time_spent": tre - trs,
        "sim_progress": (tl.time - start_t) / (stop_t - start_t) if timeout_occurred[0] else 1.0,
        "requests": [f"{src_app.node.name}-{dst_app.node.name}" for src_app, dst_app in requests],
    }
    for router in routers:
        rd = {CNT: getattr(router, CNT) for CNT in COUNTER_NAMES}
        d[router.name] = rd
    return d


if __name__ == "__main__":
    basename = f"{args.qchannel_capacity}-{args.nnodes}-{args.nedges}-{args.seed}"
    result = run_simulation(basename)
    with open(os.path.join(args.outdir, f"{basename}.json"), "w") as file:
        json.dump(result, file)
