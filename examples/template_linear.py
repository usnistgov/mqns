"""
TEMPLATE: Custom simulation script for MQNS linear-topology experiments.

Copy this file and edit the "USER CONFIG" sections.

This template allows to:
- build a linear topology S-R*-D via mqns.network.builder.NetworkBuilder
- Install a single default path S-D
- extract stats from ProactiveForwarder counters (throughput, fidelity, etc.)
- optionally compute decoherence/expired-memory metrics via LinkLayerCounters
- optionally run sweeps (parameter grids) sequentially
- save CSV/JSON and plot
"""

import itertools
import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from tap import Tap

from mqns.entity.qchannel import LinkArchDimBk, LinkArchSim, LinkArchSr
from mqns.network.builder import CTRL_DELAY, NetworkBuilder
from mqns.network.fw import SwapSequenceInput
from mqns.network.proactive import ProactiveForwarder
from mqns.network.protocol.link_layer import LinkLayerCounters
from mqns.simulator import Simulator
from mqns.utils import log, rng

from examples_common.plotting import plt, plt_save

_ = (LinkArchDimBk, LinkArchSim, LinkArchSr)


# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG: Logging
# ──────────────────────────────────────────────────────────────────────────────
# Log level can be changed via MQNS_LOGLVL environment variable.
# Use "DEBUG" while developing, "CRITICAL" to suppress most output.
log.set_default_level("CRITICAL")


# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG: Experiment defaults
# ──────────────────────────────────────────────────────────────────────────────

# Reproducibility
SEED_BASE = 100  # each run uses SEED_BASE + run_index

# Simulation controls
SIM_DURATION = 3.0  # Duration of one simulation run in seconds
SIM_ACCURACY = 1_000_000  # simulator accuracy

# Number of trials per parameter point (used in sweeps)
DEFAULT_RUNS = 10  #  Can be changed via --runs flag


# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG: Topology (linear path)
# ──────────────────────────────────────────────────────────────────────────────
# nodes:
#   - int: NetworkBuilder will auto-name ["S", "R1", ..., "D"]
#   - list[str]: explicit names (must include "S" and "D")
NODES: int | list[str] = 4

# channel_length:
#   - float: uniform length for every link
#   - list[float]: per-link lengths (must have n_links = len(nodes)-1 values)
CHANNEL_LENGTH: float | list[float] = [32.0, 18.0, 10.0]

# channel_capacity:
#   - int: uniform capacity for every link (left == right == capacity)
#   - list[int]: per-link capacity (left == right for each link)
#   - list[tuple[int,int]]: per-link (left,right) endpoint allocation
#
# NetworkBuilder interprets (left,right) as:
#   - capacity1 = allocation at node i
#   - capacity2 = allocation at node i+1
CHANNEL_CAPACITY: int | list[int] | list[tuple[int, int]] = 3

# mem_capacity:
#   - None: derived from channel_capacity (recommended for allocation-consistent setups)
#   - int: uniform #qubits per node
#   - list[int]: per-node #qubits
MEM_CAPACITY: int | list[int] | None = None

# Memory coherence time (seconds) used when SWEEP = False
T_COHERE = 0.01

# link_arch:
#   - pass a LinkArch instance (broadcast to all links)
#   - pass list[LinkArch] (per-link)
#
# If you want custom architectures, uncomment and edit:
# LINK_ARCH = [LinkArchSr(), LinkArchSim(), ...]
LINK_ARCH = [LinkArchDimBk(), LinkArchDimBk(), LinkArchDimBk()]


# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG: Physics / Link-Layer parameters (passed into NetworkBuilder)
# ──────────────────────────────────────────────────────────────────────────────
ENTG_ATTEMPT_RATE = 50e6  # attempts/sec
INIT_FIDELITY = 0.99  # fidelity of generated elementary entanglement
FIBER_ALPHA = 0.2  # dB/km
ETA_D = 0.95  # detector efficiency
ETA_S = 0.95  # source efficiency
FREQUENCY = 1e6  # entanglement source / memory frequency


# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG: Swapping / routing (passed into NetworkBuilder)
# ──────────────────────────────────────────────────────────────────────────────
# swap:
#   - preset string:
#       - 1 router: "asap"
#       - 2 to 5 routers: "asap", "l2r", "r2l", "baln"
#   - explicit list[int] sequence (for custom swap order) [see REDiP for syntax]
SWAP: SwapSequenceInput = "l2r"

# p_swap:
#   - Swapping success probability used by ProactiveForwarder(ps=p_swap)
P_SWAP = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG: Parameter sweep (optional)
# ──────────────────────────────────────────────────────────────────────────────
# If SWEEP=False, run a single scenario.
# If SWEEP=True, run a cartesian product grid sequentially.
SWEEP = True

# Supported sweep variables:
t_cohere_values = [0.005, 0.01, 0.02]  # seconds
swap_values: list[SwapSequenceInput] = ["l2r", "r2l", "asap"]  # see SWAP
channel_capacity_values = [CHANNEL_CAPACITY]  # include alternative allocations if desired


# What to measure:
type MetricName = Literal["throughput", "mean_fidelity", "expired_ratio"]
MEASURES: list[MetricName] = ["throughput", "mean_fidelity", "expired_ratio"]


# ──────────────────────────────────────────────────────────────────────────────
# Core simulation runner
# ──────────────────────────────────────────────────────────────────────────────
def run_simulation(
    *,
    seed: int,
    t_cohere: float,
    swap: SwapSequenceInput,
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
    rng.reseed(seed)

    net = (
        NetworkBuilder()
        .topo_linear(
            nodes=nodes,
            mem_capacity=MEM_CAPACITY,
            t_cohere=t_cohere,
            channel_length=channel_length,
            channel_capacity=channel_capacity,
            fiber_alpha=FIBER_ALPHA,
            link_arch=LINK_ARCH,
            entg_attempt_rate=ENTG_ATTEMPT_RATE,
            init_fidelity=INIT_FIDELITY,
            eta_d=ETA_D,
            eta_s=ETA_S,
            frequency=FREQUENCY,
            p_swap=P_SWAP,
        )
        .proactive_centralized()
        .path("S-D", swap=swap)
        .make_network()
    )

    # Run simulator for SIM_DURATION + time to install paths.
    s = Simulator(0, SIM_DURATION + CTRL_DELAY, accuracy=SIM_ACCURACY, install_to=(log, net))
    s.run()

    # ── Extract metrics ───────────────────────────────────────────────────────
    out: dict[str, float] = {}

    fw_s = net.get_node("S").get_app(ProactiveForwarder)
    e2e_count = fw_s.cnt.n_consumed

    if "throughput" in MEASURES:
        out["throughput_eps"] = e2e_count / SIM_DURATION

    if "mean_fidelity" in MEASURES:
        out["mean_fidelity"] = float(fw_s.cnt.consumed_avg_fidelity)

    if "expired_ratio" in MEASURES:
        ll_cnt = LinkLayerCounters.aggregate(net.nodes)
        out["expired_ratio"] = float(ll_cnt.decoh_ratio)
        out["expired_per_e2e_safe"] = float(ll_cnt.n_decoh / e2e_count) if e2e_count > 0 else 0.0

    return out


def run_row(
    *,
    n_runs: int,
    t_cohere: float,
    swap: SwapSequenceInput,
    channel_capacity: int | list[int] | list[tuple[int, int]],
) -> dict[str, Any]:
    """
    Run n_runs trials for one parameter point and return mean/std + raw lists.
    """
    metrics_per_run: list[dict[str, float]] = []

    for i in range(n_runs):
        seed = SEED_BASE + i
        print(f"t_cohere={t_cohere:.6f}, swap={swap}, cap={channel_capacity}, run {i + 1}/{n_runs}")

        m = run_simulation(
            seed=seed,
            t_cohere=t_cohere,
            swap=swap,
            channel_capacity=channel_capacity,
        )
        metrics_per_run.append(m)

    row: dict[str, Any] = {
        "t_cohere": t_cohere,
        "swap": str(swap),
        "channel_capacity": str(channel_capacity),
        "n_runs": n_runs,
    }

    if not metrics_per_run:
        return row

    for k in metrics_per_run[0].keys():
        vals = [d[k] for d in metrics_per_run]
        row[f"{k}_mean"] = float(np.mean(vals))
        row[f"{k}_std"] = float(np.std(vals))
        row[f"{k}_all"] = vals

    return row


def save_results(
    rows: list[dict[str, Any]],
    *,
    save_csv: str = "",
    save_json: str = "",
    save_plt: str = "",
) -> None:
    """
    Save results and plot throughput and fidelity vs coherence time.
    Plots are always shown; saving is optional.
    """
    df = pd.DataFrame(rows)

    if save_csv:
        df.to_csv(save_csv, index=False)

    if save_json:
        with open(save_json, "w") as f:
            json.dump(rows, f)

    has_throughput = "throughput_eps_mean" in df.columns
    has_fidelity = "mean_fidelity_mean" in df.columns

    if not (has_throughput or has_fidelity):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # ── Throughput subplot ────────────────────────────────────────────────
    if has_throughput:
        ax = axes[0]
        for swap_label, sub in df.groupby("swap"):
            subb = sub.sort_values("t_cohere")
            ax.errorbar(
                subb["t_cohere"],
                subb["throughput_eps_mean"],
                yerr=subb.get("throughput_eps_std", None),
                fmt="o--",
                capsize=4,
                label=swap_label,
            )
        ax.set_xscale("log")
        ax.set_xlabel("t_cohere (s)")
        ax.set_ylabel("Throughput (entanglements/sec)")
        ax.set_title("E2E Throughput")
        ax.grid(True, which="both", ls="--", lw=0.5)
        ax.legend()

    else:
        axes[0].axis("off")

    # ── Fidelity subplot ──────────────────────────────────────────────────
    if has_fidelity:
        ax = axes[1]
        for swap_label, sub in df.groupby("swap"):
            subb = sub.sort_values("t_cohere")
            ax.errorbar(
                subb["t_cohere"],
                subb["mean_fidelity_mean"],
                yerr=subb.get("mean_fidelity_std", None),
                fmt="s--",
                capsize=4,
                label=swap_label,
            )
        ax.set_xscale("log")
        ax.set_xlabel("t_cohere (s)")
        ax.set_ylabel("Mean Fidelity")
        ax.set_title("E2E Fidelity")
        ax.grid(True, which="both", ls="--", lw=0.5)
        ax.legend()

    else:
        axes[1].axis("off")

    plt.tight_layout()

    plt_save(save_plt)


# ──────────────────────────────────────────────────────────────────────────────
# CLI + Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    class Args(Tap):
        # Trials
        runs: int = DEFAULT_RUNS  # number of trials per parameter point

        # Outputs
        csv: str = ""  # optional CSV output file
        json: str = ""  # optional JSON output file
        plt: str = ""  # optional plot output file

    args = Args().parse_args()

    if not SWEEP:
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

        rows = [{"t_cohere": T_COHERE, "swap": str(SWAP), "channel_capacity": str(CHANNEL_CAPACITY), **metrics}]
        save_results(rows, save_csv=args.csv, save_json=args.json, save_plt=args.plt)

    else:
        # Sequential parameter grid sweep
        sweep_points = list(itertools.product(t_cohere_values, swap_values, channel_capacity_values))

        rows: list[dict[str, Any]] = []
        for t, sw, cap in sweep_points:
            row = run_row(n_runs=args.runs, t_cohere=t, swap=sw, channel_capacity=cap)
            rows.append(row)

        save_results(rows, save_csv=args.csv, save_json=args.json, save_plt=args.plt)

        # Quick console summary
        df = pd.DataFrame(rows)
        # Throughput summary
        if "throughput_eps_mean" in df.columns:
            cols = [
                "t_cohere",
                "swap",
                "channel_capacity",
                "throughput_eps_mean",
                "throughput_eps_std",
            ]
            print("\nTop results by throughput:")
            print(df[cols].sort_values("throughput_eps_mean", ascending=False).head(10).to_string(index=False))

        # Fidelity summary
        if "mean_fidelity_mean" in df.columns:
            cols = [
                "t_cohere",
                "swap",
                "channel_capacity",
                "mean_fidelity_mean",
                "mean_fidelity_std",
            ]
            print("\nTop results by fidelity:")
            print(df[cols].sort_values("mean_fidelity_mean", ascending=False).head(10).to_string(index=False))
