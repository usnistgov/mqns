"""
Simulate a linear quantum repeater network to validate model-driven wait-time budgets.

This script demonstrates how to configure ``CutoffSchemeWaitTime`` constraints dynamically
across a 3-node linear topology (``S``, ``R``, ``D``) using wait-time budgets from mathematical
models. The core objective is to enforce target minimum end-to-end entanglement
fidelities (``F_req``) and contrast the empirical simulation output against closed-form
theoretical model predictions.

The simulation executes in a two-stage sequential pipeline:

1. Calibration Phase (``run_calibration``)
   Executes a baseline simulation without entanglement swapping to calculate the steady-state
   elementary LinkLayer arrival rates (``lam``) across individual quantum fiber links.

2. Evaluation Phase (``run_evaluation``)
   Executes a proactive, centralized simulation using active time-out windows (``swap_cutoff``)
   at the intermediate repeater. Using the calibrated link arrival rates, it maps empirical
   results directly alongside mathematical model estimations for both throughput and fidelity.

Collected Statistics and Outputs:

* **End-to-End Entanglement Rate:** Measured in entanglements per second (Hz).
* **Fidelity Metrics:** Tracked as a macro-average across multiple stochastic runs.
  Runs with zero end-to-end deliveries are automatically isolated via ``np.nanmean`` to prevent
  skewing the average.

The script aggregates results into a multi-tiered dictionary format structured as a ``Report`` schema,
which saves directly to a JSON payload (``--json``) and compiles a dual-axis line plot
(``--plt``) mapping the throughput-fidelity tradeoff curves.
"""

import itertools
import json
from collections.abc import Sequence
from multiprocessing import Pool, freeze_support
from typing import Literal, TypedDict, override

import numpy as np
from tap import Tap

from mqns.entity.qchannel import LinkArchDimDual
from mqns.models.epr import MixedStateEntanglement
from mqns.network.builder import CTRL_DELAY, ChannelParam, NetworkBuilder, NodeDef, tap_configure
from mqns.network.network import QuantumNetwork
from mqns.network.proactive import MuxSchemeBufferSpace
from mqns.network.protocol.consumer import RequestCounters
from mqns.network.protocol.link_layer import LinkLayer
from mqns.simulator import Simulator
from mqns.utils import log, rng, seed_seq_env, unwrap

import examples_common.swap_2links as s2q_model
from examples_common.plotting import Axes, plt, plt_save

log.set_default_level("CRITICAL")


class Args(Tap):
    workers: int = 1  # number of workers for parallel execution
    runs: int = 10  # number of trials per parameter set
    sim_duration: tuple[float, float] = (5.0, 25.0)  # calibration and evaluation duration in seconds
    L: tuple[float, float] = (32, 18)  # link length
    M: tuple[int, int] = (1, 1)  # channel capacity
    gam: tuple[float, float, float] = (5, 10, 10)  # memory decohere rates in Hz (inverse of t_cohere)
    q: float = 0.5  # probability of successful swap
    t_proc: float = 0.000010  # local processing time in seconds
    depol: tuple[float, float] = (0.03, 0.02)  # optical depolarization component
    ssq: Literal["random", "oldest", "newest"] = "random"  # how to select swap qubit
    json: str = ""  # save report as JSON file
    plt: str = ""  # save plot as image file

    @override
    def configure(self) -> None:
        tap_configure(self)

    def to_json(self) -> dict:
        """Summarize input parameters to include in JSON output."""
        return {
            "runs": self.runs,
            "sim_duration": self.sim_duration,
            "L": self.L,
            "M": self.M,
            "gam": self.gam,
            "t_proc": self.t_proc,
            "depol": self.depol,
            "ssq": self.ssq,
        }

    def to_matching_policy(self) -> s2q_model.MatchingPolicy:
        """Convert to swap_2Q matching_policy parameter."""
        match self.ssq:
            case "random":
                return "RANDOM"
            case "oldest":
                return "FIFO"
            case "newest":
                return "LIFO"


SIMULATOR_ACCURACY = 1000000
F_REQ_VALUES: Sequence[float] = [0.75, 0.7823, 0.8146, 0.8469, 0.8791, 0.8904, 0.9001, 0.9098]
"""Required fidelity values."""


class Stats(TypedDict):
    rate: float
    """Rate -- number of entanglements consumed per second."""
    fid_mean: float
    """Fidelity -- mean."""
    fid_std: float
    """Fidelity -- standard deviation."""


class Row(TypedDict):
    """Evaluation result for one f_req, together with model estimation."""

    f_req: float
    """Required fidelity."""
    m_budgets: dict
    """Wait-time budgets and swap timing."""
    m_bd: list[float]
    """Rate and fidelity from BD model."""
    m_ctmc: list[float]
    """Rate and fidelity from CTMC model."""
    m_sim: list[float]
    """Rate and fidelity from SIM model."""
    rate_mean: float
    """Rate -- mean."""
    rate_std: float
    """Rate -- standard deviation."""
    fid_mean: float
    """Fidelity -- mean of per-run means."""
    fid_std: float
    """Fidelity -- standard deviation of per-run means."""
    runs: list[Stats]
    """Per-run statistics."""


class Report(TypedDict):
    """JSON report."""

    args: dict
    """Input arguments summary."""
    lam_mean: list[float]
    """LinkLayer arrival rate for each link -- mean from calibration."""
    lam_std: list[float]
    """LinkLayer arrival rate for each link -- standard deviation from calibration."""
    lam_runs: list[list[float]]
    """LinkLayer arrival rate for each link -- per calibration run."""
    eval: list[Row]
    """Evaluation results."""


def _convert_fidelity(raw_error: s2q_model.PauliError) -> Sequence[float]:
    raw_fidelity, (x_ratio, y_ratio, z_ratio) = raw_error
    p_error = 1 - raw_fidelity
    return raw_fidelity, p_error * z_ratio, p_error * x_ratio, p_error * y_ratio


def _run_simulation(
    seed: int,
    args: Args,
    duration: float,
    modeled: s2q_model.ComputedWaitTimeBudget | None,
) -> QuantumNetwork:
    rng.reseed(seed)

    b = NetworkBuilder(epr_type=MixedStateEntanglement)

    if modeled is None:
        channels = [ChannelParam(ch_length=l, ch_capacity=m) for l, m in zip(args.L, args.M, strict=True)]
    else:
        channels = [
            ChannelParam(ch_length=l, ch_capacity=m, init_fidelity=_convert_fidelity(f))
            for l, m, f in zip(args.L, args.M, modeled.pauli, strict=True)
        ]

    b.topo_linear(
        nodes=[NodeDef(node, t_cohere=1 / gam) for node, gam in zip("SRD", args.gam, strict=True)],
        frequency=80e6,
        tau_0=args.t_proc,
        channels=channels,
        link_arch=LinkArchDimDual,
        fiber_alpha=0.2,
        eta_s=0.99,
        eta_d=0.58,
    ).proactive_centralized(
        p_swap=args.q,
        swap_delay=0 if modeled is None else modeled.Tswp,
        swap_error="PERFECT",  # no error applied by the swap gates
        swap_error_at="f",  # memory decoherence continues during swap
        mux=MuxSchemeBufferSpace(select_swap_qubit=args.ssq),
    )

    if modeled is None:
        b.request("S-D", swap="disabled")
    else:
        b.request("S-D", swap_cutoff=modeled.W)

    net = b.make_network()

    RequestCounters.enable_collect_all(net, 0, "S-D")

    s = Simulator(0, duration + CTRL_DELAY, accuracy=SIMULATOR_ACCURACY, install_to=(log, net))
    s.run()

    return net


def run_calibration(seed: int, args: Args) -> list[float]:
    """
    Perform a calibration run with swapping disabled.

    Args:
        Entanglement arrival rate for each qubit pair on a link.
        The list must have ``len(args.L)`` elements.
    """
    duration = args.sim_duration[0]
    net = _run_simulation(seed, args, duration, None)

    def get_lam(node_pri: str, m: int) -> float:
        ll_cnt = net.get_node(node_pri).get_app(LinkLayer).cnt
        return ll_cnt.n_etg / duration / m

    return [get_lam(node, m) for node, m in zip("SR", args.M, strict=True)]


def run_evaluation(seed: int, args: Args, modeled: s2q_model.ComputedWaitTimeBudget) -> Stats:
    """
    Perform an evaluation run with swapping enabled.

    Args:
        modeled: Wait-time budgets and swap timing.
    """
    duration = args.sim_duration[1]
    net = _run_simulation(seed, args, duration, modeled)

    req_cnt = RequestCounters.of(net, 0, "S-D")
    rate = req_cnt.get_rate(duration)
    fids = np.array(unwrap(req_cnt.consumed_fidelity_values), dtype=float)
    if len(fids) == 0:
        return Stats(rate=rate, fid_mean=0, fid_std=0)
    return Stats(rate=rate, fid_mean=fids.mean(), fid_std=fids.std())


def _make_queue_spec(m: int, lam1: float, w: float, t: float) -> s2q_model.QueueSpec:
    lam = [lam1 * (m - n) for n in range(m + 1)]
    return s2q_model.QueueSpec(M=m, lam=lam, tau=None, W=w, T=t, f=None)


def run_row(args: Args, f_req: float, lam_mean: list[float]) -> Row:
    modeled = s2q_model.compute_wait_time_budgets(
        F_req=f_req,
        L=args.L,
        lam=lam_mean,
        gam=args.gam,
        T_proc=args.t_proc,
        q=args.q,
        depol=args.depol,
    )

    runs = [run_evaluation(seed, args, modeled) for seed in seed_seq_env(args.runs, 100)]

    rates = np.fromiter((s["rate"] for s in runs), dtype=float)
    fids = np.fromiter((s["fid_mean"] if s["rate"] > 0 else np.nan for s in runs), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fid_mean, fid_std = np.nanmean(fids).item(), np.nanstd(fids).item()

    m_budgets = modeled._asdict()
    del m_budgets["F_req"]
    del m_budgets["w2t"]

    Q1, Q2 = [_make_queue_spec(*args) for args in zip(args.M, lam_mean, modeled.W, modeled.T, strict=True)]
    matching_policy = args.to_matching_policy()

    def run_model(method: s2q_model.Method):
        raw_rate, wait_times = s2q_model.swap_2Q(
            Q1,
            Q2,
            Tswp=modeled.Tswp,
            method=method,
            matching_policy=matching_policy,
        )
        return [args.q * raw_rate, modeled.w2t(wait_times)]

    return Row(
        f_req=f_req,
        m_budgets=m_budgets,
        m_bd=run_model("BD"),
        m_ctmc=run_model("CTMC"),
        m_sim=run_model("SIM"),
        rate_mean=rates.mean(),
        rate_std=rates.std(),
        fid_mean=fid_mean,
        fid_std=fid_std,
        runs=runs,
    )


def main(args: Args) -> Report:
    # Run calibration step to determine LinkLayer arrival rates of each channel.
    with Pool(processes=args.workers) as pool:
        lam_runs = pool.starmap(run_calibration, itertools.product(seed_seq_env(args.runs, 100), [args]))
    lam_mean: list[float] = []
    lam_std: list[float] = []
    # Collect mean LinkLayer arrival rates of each channel.
    for i in range(len(args.L)):
        lam_array = np.fromiter((lam_run[i] for lam_run in lam_runs), dtype=float)
        lam_mean.append(lam_array.mean())
        lam_std.append(lam_array.std())

    # Run evaluation step to determine end-to-end rate and fidelity.
    # Theoretical rate and fidelity are also calculated based on desired fidelity and calibrated LinkLayer arrival rates.
    with Pool(processes=args.workers) as pool:
        rows = pool.starmap(run_row, itertools.product([args], F_REQ_VALUES, [lam_mean]))

    # Generate report.
    return Report(
        args=args.to_json(),
        lam_mean=lam_mean,
        lam_std=lam_std,
        lam_runs=lam_runs,
        eval=rows,
    )


def plot(args: Args, rows: list[Row]) -> None:
    F_req = [row["f_req"] for row in rows]

    def plot_model_rate_or_fid(ax: Axes, color: str, i: int, prefix: str) -> None:
        ax.plot(F_req, [row["m_bd"][i] for row in rows], linestyle="solid", color=color, label=f"{prefix} (BD)")
        ax.plot(F_req, [row["m_ctmc"][i] for row in rows], linestyle="dashed", color=color, label=f"{prefix} (CTMC)")
        ax.plot(F_req, [row["m_sim"][i] for row in rows], linestyle="dotted", color=color, label=f"{prefix} (SIM)")

    fig, ax1 = plt.subplots(figsize=(5, 4), constrained_layout=True)
    color = "tab:blue"
    ax1.set_xlabel("Required Fidelity")
    ax1.set_ylabel("Rate (epps)", color=color)
    plot_model_rate_or_fid(ax1, color, 0, "Rate")
    ax1.errorbar(
        F_req,
        [row["rate_mean"] for row in rows],
        yerr=[row["rate_std"] for row in rows],
        fmt="d",
        color=color,
        label="Simulated rate",
        markersize=4,
        capsize=3,
    )
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Fidelity", color=color)
    plot_model_rate_or_fid(ax2, color, 1, "Fidelity")
    ax2.errorbar(
        F_req,
        [row["fid_mean"] for row in rows],
        yerr=[row["fid_std"] for row in rows],
        fmt="d",
        color=color,
        label="Simulated fidelity",
        markersize=4,
        capsize=3,
    )
    ax2.tick_params(axis="y", labelcolor=color)

    fig.suptitle("Fidelity vs Rate tradeoff", fontsize="medium")
    fig.legend(loc="center left", bbox_to_anchor=(0.12, 0.5))
    plt.grid()
    plt_save(args.plt)


if __name__ == "__main__":
    freeze_support()
    args = Args().parse_args()
    report = main(args)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file)
    plot(args, report["eval"])
