import json
import os.path

from mqns.entity.node import QNode
from mqns.network.network import Request
from mqns.network.proactive import (
    LinkLayer,
    ProactiveForwarder,
    ProactiveRoutingController,
    QubitAllocationType,
    RoutingPathSingle,
)
from mqns.simulator import Simulator
from mqns.utils import WallClockTimeout, json_default, log

from examples_common.scalability_randomtopo import RequestStats, RunArgs, RunResult, build_network, parse_run_args

"""
This script is typically invoked as part of scalability_randomtopo experiment.
See scalability_randomtopo.sh for how to run this script.
"""

log.set_default_level("CRITICAL")


def run_simulation(args: RunArgs) -> RunResult:
    # Generate random topology and requests.
    # Random seed is set within.
    net = build_network(args)

    # Install network into Simulator.
    s = Simulator(0, args.sim_duration + 5e-06, accuracy=1000000, install_to=(log, net))

    # Install paths for requests.
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    for req in net.requests:
        ctrl.install_path(
            RoutingPathSingle(req.src.name, req.dst.name, qubit_allocation=QubitAllocationType.DISABLED, swap="asap")
        )

    # Run the simulation.
    timeout = WallClockTimeout(args.time_limit, stop=s.stop)
    with timeout():
        s.run()
    sim_duration = s.tc.sec if timeout.occurred else args.sim_duration

    # Collect results.
    def gather_request_stats(req: Request) -> RequestStats:
        fw = req.src.get_app(ProactiveForwarder)
        return fw.cnt.n_consumed / sim_duration, fw.cnt.consumed_avg_fidelity

    def gather_node_stats(node: QNode):
        fw = node.get_app(ProactiveForwarder)
        ll = node.get_app(LinkLayer)
        return [
            ll.cnt,
            fw.cnt,
        ]

    return RunResult(
        time_spent=s.time_spend,
        sim_progress=sim_duration / args.sim_duration,
        requests={f"{req.src.name}-{req.dst.name}": gather_request_stats(req) for req in net.requests},
        nodes={node.name: gather_node_stats(node) for node in net.nodes},
    )


if __name__ == "__main__":
    args = parse_run_args()
    result = run_simulation(args)
    with open(os.path.join(args.outdir, f"{args.basename}.json"), "w") as file:
        json.dump(result, file, default=json_default)
