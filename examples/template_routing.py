"""
TEMPLATE: CustomTopology + routing + multiplexing, with a fixed coherence-time sweep.

This script runs a small proactive-entanglement simulation on a single, reusable custom topology
to demonstrate three common tutorial scenarios:

1) Two simultaneous end-to-end requests (S1->D1 and S2->D2) competing for a shared trunk (R2<->R3)
   - You can run this with Statistical multiplexing (no explicit qubit pre-allocation), or
   - Buffer-Space multiplexing with an explicit per-hop MultiplexingVector (manual allocation).

2) A single request with multiple routing alternatives (multipath) for S1->D1
   - The topology contains two distinct routes from S1 to D1 (via R1 or via R5).
   - Yen's k-shortest-paths algorithm is used to provide multiple candidate paths.

For each scenario, the script sweeps the memory coherence time `t_cohere`.

For every sweep point, the script runs `--runs` runs (different RNG seeds) and reports the
mean ± std end-to-end entanglement consumption rate (eps). It also records mean fidelity for each measured flow.

How to use it:
1) Pick a scenario:
   Set `ACTIVE_SCENARIO_KEY` to one of:
     - "one_flow_buffer_space_single"
     - "two_flows_statistical_single"
     - "two_flows_buffer_space_single"
     - "single_request_multipath"

2) (Optional) Customize routing + multiplexing:
   Edit the corresponding entry in the `SCENARIOS` dict:
     - `install_paths`: what the controller installs (RoutingPathSingle / RoutingPathStatic / RoutingPathMulti)
     - `mux`: which multiplexing scheme is used (Statistical or Buffer-Space here)
     - `route_algorithm`: which routing algorithm is used (Dijkstra or Yen)
     - `measured_sources`: which source nodes to read counters from for reporting

3) (Optional) Customize the topology:
   Edit `TOPO_SPEC` to change:
     - nodes and links
     - link lengths and capacities (e.g., to make the trunk more/less contested)

4) Run:
   python your_script.py --runs 5
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from tap import Tap

from mqns.network.builder import CTRL_DELAY, NetworkBuilder
from mqns.network.network import QuantumNetwork
from mqns.network.network.timing import TimingModeSync
from mqns.network.proactive import (
    MuxScheme,
    MuxSchemeBufferSpace,
    MuxSchemeStatistical,
    ProactiveForwarder,
    RoutingPath,
    RoutingPathMulti,
    RoutingPathSingle,
    RoutingPathStatic,
)
from mqns.network.proactive.message import MultiplexingVector
from mqns.network.proactive.routing import QubitAllocationType
from mqns.network.route import DijkstraRouteAlgorithm, YenRouteAlgorithm
from mqns.simulator import Simulator
from mqns.utils import log, rng

_ = TimingModeSync


# =============================================================================
# CLI
# =============================================================================
class Args(Tap):
    runs: int = 3


args = Args().parse_args()
log.set_default_level("CRITICAL")


# =============================================================================
# Fixed simulation parameters (users can edit these)
# =============================================================================
SEED_BASE = 100
SIM_DURATION = 3.0

FIBER_ALPHA_DB_PER_KM = 0.2
ETA_D = 0.95
ETA_S = 0.95
MEMORY_FREQUENCY = 1e6
ENTG_ATTEMPT_RATE = 50e6  # if your LinkLayer uses attempt_rate
INIT_FIDELITY = 0.99
P_SWAP = 0.5

# SWAP-ASAP swapping policy is preferable as it works on all scenarios.
SWAPPING_POLICY = "asap"


# =============================================================================
# SWEEP memory coherence time (t_cohere)
# =============================================================================
T_COHERE_SWEEP = [0.1, 0.01, 0.001]  # seconds


# =============================================================================
# Custom topology that supports:
#   - two simultaneous requests (S1->D1, S2->D2)
#   - single-request multipath (S1->D1 has two paths)
#   - shared trunk contention (R2<->R3 is shared)
#
# Graph (quantum + classical share the same connectivity):
#
#     S1 -- R5
#      \     \
#       R1 -- R2 ===== R3 -- R4 -- D1
#              \        \
#               \        +--- D2
#                \
#                 S2
# =============================================================================

# Link lengths
L_S1_R1 = 10
L_S1_R5 = 15
L_R5_R2 = 15
L_R1_R2 = 10
L_S2_R2 = 10
L_R2_R3 = 10  # contested trunk
L_R3_R4 = 10
L_R4_D1 = 10
L_R3_D2 = 10

# Capacities (per link endpoint).
CAP_DEFAULT = 2


# =============================================================================
# Use a Scenario object to define:
#   - which paths get installed by the controller
#   - which multiplexing scheme is used
#
# Users: pick one option below by setting ACTIVE_SCENARIO_KEY.
# Then only edit within this SCENARIOS dict if you want custom behavior.
#
# Notes:
# - Statistical mux: best-effort, no manual allocation.
# - Buffer-space mux: you can pass explicit MultiplexingVectors in RoutingPathStatic.
# - Single-request multipath: uses RoutingPathMulti + YenRouteAlgorithm.
# =============================================================================
@dataclass(frozen=True)
class Scenario:
    install_paths: list[RoutingPath]
    mux: MuxScheme
    route_algorithm: Any
    measured_sources: list[str]  # which nodes we read ProactiveForwarder counters from
    measured_labels: list[str]  # labels aligned with measured_sources (for plotting)


def _mux_statistical() -> MuxScheme:
    # select_swap_qubit: Function to select a qubit to swap with, default is first.
    # select_path: Function to select a FIB entry for signaling after swap, default is random.
    return MuxSchemeStatistical(
        select_swap_qubit=MuxSchemeStatistical.SelectSwapQubit_random, select_path=MuxSchemeStatistical.SelectPath_random
    )


def _mux_buffer_space() -> MuxScheme:
    # select_swap_qubit: Function to select a qubit to swap with, default is first.
    return MuxSchemeBufferSpace(select_swap_qubit=MuxSchemeBufferSpace.SelectSwapQubit_random)


# Multiplexing vector for buffer-space mux. It's per-hop list of (tx_qubits, rx_qubits) allocations.
# This example gives each flow half of the trunk budget.
def _mv_two_flows_equal_share(route: list[str]) -> MultiplexingVector:
    mv: MultiplexingVector = []
    for u, v in zip(route[:-1], route[1:]):
        if (u, v) == ("R2", "R3") or (u, v) == ("R3", "R2"):
            mv.append((max(1, CAP_DEFAULT // 2), max(1, CAP_DEFAULT // 2)))
        else:
            mv.append((CAP_DEFAULT, CAP_DEFAULT))
    return mv


# Static explicit routes for the two flows (you can edit these hop lists)
ROUTE_S1_D1 = ["S1", "R1", "R2", "R3", "R4", "D1"]
ROUTE_S2_D2 = ["S2", "R2", "R3", "D2"]


SCENARIOS: dict[str, Scenario] = {
    # ------------------------------------------------------------
    # One request, single-path (Similar to Turorial 1)
    # Buffer-space mux --> follow quantum channel capacity
    # ------------------------------------------------------------
    "one_flow_buffer_space_single": Scenario(
        # let routing algorithm compute paths
        install_paths=[RoutingPathSingle("S1", "D1", swap=SWAPPING_POLICY)],
        mux=_mux_buffer_space(),
        route_algorithm=DijkstraRouteAlgorithm(),
        measured_sources=["S1"],
        measured_labels=["S1-D1"],
    ),
    # ------------------------------------------------------------
    # Two simultaneous requests, single-path per request
    # Statistical mux --> disable Qubit pre-allocation
    # ------------------------------------------------------------
    "two_flows_statistical_single": Scenario(
        # let routing algorithm compute paths
        install_paths=[
            RoutingPathSingle("S1", "D1", qubit_allocation=QubitAllocationType.DISABLED, swap=SWAPPING_POLICY),
            RoutingPathSingle("S2", "D2", qubit_allocation=QubitAllocationType.DISABLED, swap=SWAPPING_POLICY),
        ],
        mux=_mux_statistical(),
        route_algorithm=DijkstraRouteAlgorithm(),
        measured_sources=["S1", "S2"],
        measured_labels=["S1-D1", "S2-D2"],
    ),
    # ------------------------------------------------------------
    # Two simultaneous requests, single-path per request
    # Buffer-space multiplexing with a manual MultiplexingVector split on trunk R2-R3
    # ------------------------------------------------------------
    "two_flows_buffer_space_single": Scenario(
        # manually set paths + qubit allocation
        install_paths=[
            RoutingPathStatic(ROUTE_S1_D1, m_v=_mv_two_flows_equal_share(ROUTE_S1_D1), swap=SWAPPING_POLICY),
            RoutingPathStatic(ROUTE_S2_D2, m_v=_mv_two_flows_equal_share(ROUTE_S2_D2), swap=SWAPPING_POLICY),
        ],
        mux=_mux_buffer_space(),
        route_algorithm=DijkstraRouteAlgorithm(),
        measured_sources=["S1", "S2"],
        measured_labels=["S1-D1", "S2-D2"],
    ),
    # ------------------------------------------------------------
    # Single request, multipath
    # Use YenRouteAlgorithm
    # Only compatible with buffer-space multuplexing. The qubits are divided among all paths that share the qchannel.
    # ------------------------------------------------------------
    "single_request_multipath": Scenario(
        install_paths=[RoutingPathMulti("S1", "D1", swap=SWAPPING_POLICY)],
        mux=_mux_buffer_space(),
        route_algorithm=YenRouteAlgorithm(k_paths=2),  # number of paths. Default is 3.
        measured_sources=["S1"],
        measured_labels=["S1-D1 (multipath)"],
    ),
}

# Pick ONE scenario here:
ACTIVE_SCENARIO_KEY = "single_request_multipath"
SC = SCENARIOS[ACTIVE_SCENARIO_KEY]


# =============================================================================
# Build Network
# =============================================================================
def build_network(route_algo: Any, t_cohere: float) -> QuantumNetwork:
    # b = NetworkBuilder(route=route_algo, timing=TimingModeSync(t_ext=t_cohere / 2, t_int=t_cohere / 2))
    b = NetworkBuilder(route=route_algo)

    b.topo(
        channels=[
            (("S1", "R1"), L_S1_R1, CAP_DEFAULT),
            (("S1", "R5"), L_S1_R5, CAP_DEFAULT),
            (("R5", "R2"), L_R5_R2, CAP_DEFAULT),
            (("R1", "R2"), L_R1_R2, CAP_DEFAULT),
            (("S2", "R2"), L_S2_R2, CAP_DEFAULT),
            (("R2", "R3"), L_R2_R3, CAP_DEFAULT),
            (("R3", "R4"), L_R3_R4, CAP_DEFAULT),
            (("R4", "D1"), L_R4_D1, CAP_DEFAULT),
            (("R3", "D2"), L_R3_D2, CAP_DEFAULT),
        ],
        fiber_alpha=FIBER_ALPHA_DB_PER_KM,
        eta_d=ETA_D,
        eta_s=ETA_S,
        entg_attempt_rate=ENTG_ATTEMPT_RATE,
        frequency=MEMORY_FREQUENCY,
        init_fidelity=INIT_FIDELITY,
        p_swap=P_SWAP,
        t_cohere=t_cohere,
    )

    b.proactive_centralized(mux=SC.mux)

    for flow in SC.install_paths:
        b.path(flow)

    return b.make_network()


# =============================================================================
# One run + metric extraction
# =============================================================================
def run_one(t_cohere: float, seed: int) -> dict[str, tuple[float, float]]:
    rng.reseed(seed)

    net = build_network(SC.route_algorithm, t_cohere)

    sim = Simulator(0, SIM_DURATION + CTRL_DELAY, accuracy=1_000_000, install_to=(log, net))
    sim.run()

    # Collect stats at selected sources (one per “request” in this tutorial pattern)
    out: dict[str, tuple[float, float]] = {}
    for src, label in zip(SC.measured_sources, SC.measured_labels):
        fw = net.get_node(src).get_app(ProactiveForwarder)
        rate = fw.cnt.n_consumed / SIM_DURATION
        fid = fw.cnt.consumed_avg_fidelity
        out[label] = (rate, fid)

    return out


# =============================================================================
# Sweep driver (mean/std over runs)
# =============================================================================
def run_sweep():
    print(f"Running Scenario: {ACTIVE_SCENARIO_KEY}")
    labels = SC.measured_labels

    # results[label]["rate_mean/std"], ["fid_mean/std"] each as list over sweep points
    results: dict[str, dict[str, list[float]]] = {
        lab: {"rate_mean": [], "rate_std": [], "fid_mean": [], "fid_std": []} for lab in labels
    }

    for t in T_COHERE_SWEEP:
        # runs: list of {label: (rate,fid)}
        runs = [run_one(t, SEED_BASE + i) for i in range(args.runs)]

        for lab in labels:
            rates = [tr[lab][0] for tr in runs]
            fids = [tr[lab][1] for tr in runs]
            results[lab]["rate_mean"].append(float(np.mean(rates)))
            results[lab]["rate_std"].append(float(np.std(rates)))
            results[lab]["fid_mean"].append(float(np.mean(fids)))
            results[lab]["fid_std"].append(float(np.std(fids)))

        print(
            f"[done] t_cohere={t:.6f}s | "
            + " | ".join(
                f"{lab}: rate={results[lab]['rate_mean'][-1]:.3f}±{results[lab]['rate_std'][-1]:.3f}" for lab in labels
            )
        )

    return results


if __name__ == "__main__":
    results = run_sweep()
