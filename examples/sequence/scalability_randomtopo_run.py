import configparser
import json
import os.path
import random
import threading
import time
from typing import cast

from sequence.network_management.network_manager import ResourceReservationProtocol
from sequence.network_management.reservation import Reservation
from sequence.topology.node import QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo
from tap import Tap

from mqns.utils import set_seed

from sequence_detail.resource_reservation import create_rules
from sequence_detail.scalability_randomtopo import (
    EntanglementRequestApp,
    Request,
    ResetApp,
    create_random_quantum_network,
    set_parameters,
)


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


def build_network(basename: str) -> tuple[RouterNetTopo, list[QuantumRouter]]:
    filename = os.path.join(args.outdir, f"{basename}.topo.json")

    create_random_quantum_network(
        num_nodes=args.nnodes,
        num_edges=args.nedges,
        edge_length=30000,
        stop_time=stop_t,
        output_file=filename,
    )
    topo = RouterNetTopo(filename)
    set_parameters(topo, common_config)
    ResourceReservationProtocol.create_rules = create_rules
    Reservation.link_capacity = args.qchannel_capacity
    Reservation.swapping_order = "ASAP"

    routers = cast(list[QuantumRouter], topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER))
    for router in routers:
        router.network_manager.network_routers = routers
        for CNT in COUNTER_NAMES:
            setattr(router, CNT, 0)

    return topo, routers


def random_requests(routers: list[QuantumRouter], num_requests: int) -> list[Request]:
    routers = routers.copy()
    random.shuffle(routers)
    requests: list[Request] = []
    for _ in range(num_requests):
        assert len(routers) >= 2
        src_node, dst_node = routers.pop(), routers.pop()
        app_src = EntanglementRequestApp(src_node, dst_node.name)
        app_dst = ResetApp(dst_node, src_node.name)
        requests.append((app_src, app_dst))
    return requests


def start_requests(requests: list[Request]) -> None:
    for app_src, app_dst in requests:
        app_src.start(app_dst.node.name, start_t, stop_t, memo_size=1, fidelity=0.1)
        app_dst.set(start_t, stop_t)


def run_simulation(basename: str) -> dict:
    # Assign random seed.
    set_seed(args.seed)

    # Generate random topology.
    topo, routers = build_network(basename)

    # Generate random requests, proportional to network size.
    num_requests = max(2, int(args.nnodes / 10))
    requests = random_requests(routers, num_requests)

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
