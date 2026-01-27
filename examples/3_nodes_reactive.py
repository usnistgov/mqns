"""
Simulate Reactive routing in a 3-node linear topology and report end-to-end throughput.
"""


from typing import Any, Literal

from mqns.network.network import TimingModeSync
from mqns.network.network import QuantumNetwork
from mqns.network.reactive import ReactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from mqns.entity.qchannel import LinkArchDimBk 

from examples_common.stats import gather_etg_decoh
from examples_common.topo_linear_reactive import build_topology

log.set_default_level("DEBUG")

SEED_BASE = 100

SIM_DURATION = 10.0         # seconds
SIM_ACCURACY = 1_000_000


# ──────────────────────────────────────────────────────────────────────────────
# Topology (linear path)
# ──────────────────────────────────────────────────────────────────────────────
# nodes:
#   - int: build_topology will auto-name ["S", "R1", ..., "D"]
#   - list[str]: explicit names (must include "S" and "D")
NODES: int | list[str] = ["S", "R", "D"]

# channel_length:
#   - float: uniform length for every link
#   - list[float]: per-link lengths (must have n_links = len(nodes)-1 values)
CHANNEL_LENGTH: float | list[float] = [32.0, 18.0]

# channel_capacity:
#   - int: uniform capacity for every link (left == right == capacity)
#   - list[int]: per-link capacity (left == right for each link)
#   - list[tuple[int,int]]: per-link (left,right) endpoint allocation
#
# build_topology interprets (left,right) as:
#   - capacity1 = allocation at node i
#   - capacity2 = allocation at node i+1
CHANNEL_CAPACITY: int | list[int] | list[tuple[int, int]] = 3

# mem_capacity:
#   - None: derived from channel_capacity (recommended for allocation-consistent setups)
#   - int: uniform #qubits per node
#   - list[int]: per-node #qubits
MEM_CAPACITY: int | list[int] | None = None

# Memory coherence time (seconds)
T_COHERE = 0.1

# link_arch:
#   - None uses build_topology default LinkArchDimBkSeq()
#   - or pass a LinkArch instance (broadcast to all links)
#   - or pass list[LinkArch] (per-link)
#
# If you want custom architectures, uncomment and edit:
# LINK_ARCH = [LinkArchSr(), LinkArchSim(), ...]
LINK_ARCH =  [LinkArchDimBk(), LinkArchDimBk()]  # None means "don't pass link_arch; use builder default"


# ──────────────────────────────────────────────────────────────────────────────
# Physics / Link-Layer parameters
# ──────────────────────────────────────────────────────────────────────────────
ENTG_ATTEMPT_RATE = 50e6   # attempts/sec
INIT_FIDELITY = 0.99       # fidelity of generated elementary entanglement
FIBER_ALPHA = 0.2          # dB/km
ETA_D = 0.95               # detector efficiency
ETA_S = 0.95               # source efficiency
FREQUENCY = 1e6            # entanglement source / memory frequency


# ──────────────────────────────────────────────────────────────────────────────
# Swapping / routing
# ──────────────────────────────────────────────────────────────────────────────
# swap:
#   - preset string:
#       - 1 router: "swap_1"
#       - 2 to 5 routers: "asap", "l2r", "r2l", "baln" 
#   - explicit list[int] sequence (for custom swap order) [see REDiP for syntax]
SWAP: str | list[int] = "swap_1"

# p_swap:
#   - Swapping success probability used by ReactiveForwarder(ps=p_swap)
P_SWAP = 1.0


# What to measure:
MetricName = Literal["throughput", "mean_fidelity", "expired_ratio"]
MEASURES: list[MetricName] = ["throughput", "mean_fidelity", "expired_ratio"]


# ──────────────────────────────────────────────────────────────────────────────
# Core simulation runner
# ──────────────────────────────────────────────────────────────────────────────
def run_simulation(
    *,
    seed: int,
    t_cohere: float,
    swap: str | list[int],
    channel_capacity: int | list[int] | list[tuple[int, int]],
    channel_length: float | list[float] = CHANNEL_LENGTH,
    nodes: int | list[str] = NODES,
) -> dict[str, float]:
    """
    Run one simulation instance and return scalar metrics.

    Customize by:
    - returning additional metrics
    - changing which node/app counters you read
    - adding your own logging or trace collection
    """
    set_seed(seed)

    topo_kwargs: dict[str, Any] = dict(
        nodes=nodes,
        mem_capacity=MEM_CAPACITY,
        t_cohere=t_cohere,
        channel_length=channel_length,
        channel_capacity=channel_capacity,
        entg_attempt_rate=ENTG_ATTEMPT_RATE,
        init_fidelity=INIT_FIDELITY,
        fiber_alpha=FIBER_ALPHA,
        eta_d=ETA_D,
        eta_s=ETA_S,
        frequency=FREQUENCY,
        p_swap=P_SWAP,
        swap=swap,
    )

    # Only pass link_arch if user configured it; otherwise builder default applies.
    if LINK_ARCH is not None:
        topo_kwargs["link_arch"] = LINK_ARCH

    topo = build_topology(**topo_kwargs)
    timing = TimingModeSync(t_ext=0.03, t_rtg=0.00005, t_int=0.0002)       # set phases durations
    net = QuantumNetwork(topo, timing=timing)           # use Synchronous timing
    net.add_request("S", "D")              # set an E2E etg. request (NOT install path) to be served by the network

    # Run simulator for SIM_DURATION + time to install paths.
    s = Simulator(0, SIM_DURATION, accuracy=SIM_ACCURACY, install_to=(log, net))
    s.run()

    # ── Extract metrics ───────────────────────────────────────────────────────
    out: dict[str, float] = {}

    fw_s = net.get_node("S").get_app(ReactiveForwarder)
    e2e_count = fw_s.cnt.n_consumed

    if "throughput" in MEASURES:
        out["throughput_eps"] = e2e_count / SIM_DURATION

    if "mean_fidelity" in MEASURES:
        out["mean_fidelity"] = float(fw_s.cnt.consumed_avg_fidelity)

    if "expired_ratio" in MEASURES:
        _, total_decohered, decoh_ratio = gather_etg_decoh(net)
        out["expired_ratio"] = float(decoh_ratio)
        out["expired_per_e2e_safe"] = float(total_decohered / e2e_count) if e2e_count > 0 else 0.0

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Single scenario run
    metrics = run_simulation(
        seed=SEED_BASE,
        t_cohere=T_COHERE,
        swap=SWAP,
        channel_capacity=CHANNEL_CAPACITY,
    )
    print("Single-run metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")