"""
Simulate a multi-tenant quantum network to evaluate multiplexing strategies.

This script benchmarks different link-sharing methodologies across a 13-node tree/star
topology characterized by two highly contested central backbone routing links: ``E-F`` and ``F-J``.

.. figure:: /_static/examples/multiplexing.svg
   :alt: topology diagram
   :align: center
   :width: 100%

Five competing end-to-end data flows share segments of this backbone infrastructure:

* ``AK``: A -> E -> F -> J -> K
* ``BL``: B -> E -> F -> J -> L
* ``CI``: C -> E -> F -> I
* ``DH``: D -> E -> F -> H
* ``GM``: G -> F -> J -> M

Evaluated Multiplexing Strategies (``STRATEGIES``):

1.  **Statistical Multiplexing (``MuxSchemeStatistical``):** Flows access link-level entanglement on a competitive,
    best-effort basis without advance hardware reservations.
2.  **Buffer-Space Multiplexing (``MuxSchemeBufferSpace``):** Enforces a strict, proactive partitioning of
    available channel memories per active flow along contested trunks.

Experimental Protocol:
The script systematically executes a matrix of multi-flow loading scenarios (from isolated
single flows up to full 5-way traffic saturation) across both routing schemes.
It collects source-destination throughput rates (eps), state fidelities, memory decoherence
counts, and swap scheduling contentions.

Outputs:

* **JSON Report (``--json``):** Scenario metrics per run including rate and fidelity.
* **Stacked Bar Charts (``--plt_stat``, ``--plt_buff``):** Visual summaries breaking down
  aggregate network throughput by individual flow contributions relative to uncontested baselines.
"""

import itertools
import json
from collections.abc import Sequence
from multiprocessing import Pool, freeze_support
from typing import NamedTuple

import numpy as np
from tap import Tap

from mqns.network.builder import CTRL_DELAY, NetworkBuilder
from mqns.network.fw import MultiplexingVector, RoutingPath
from mqns.network.ll import LinkLayer
from mqns.network.network import QuantumNetwork
from mqns.network.proactive import (
    MuxSchemeBufferSpace,
    MuxSchemeInput,
    MuxSchemeStatistical,
    ProactiveForwarder,
    mux_scheme_is_buffer_space,
)
from mqns.network.protocol.consumer import RequestCounters
from mqns.simulator import Simulator
from mqns.utils import log, rng, seed_seq_env

from examples_common.plotting import mpl, plt, plt_save
from examples_common.topo_multiplexing import define_topo

log.set_default_level("CRITICAL")


class Args(Tap):
    workers: int = 1  # number of workers for parallel execution
    runs: int = 3  # number of trials per parameter set
    sim_duration: float = 3.0  # simulation duration in seconds
    json: str = ""  # save results as JSON file
    plt_stat: str = ""  # save Statistical plot as image file
    plt_buff: str = ""  # save Buffer-Space plot as image file


class FlowDef:
    def __init__(self, req_id: int, route: Sequence[str], color: str):
        self.req_id = req_id
        self.route = list(route)
        self.label = f"{self.route[0]}{self.route[-1]}"
        self.color = color
        self.idx = -1


TX_QUBITS = 50
RX_QUBITS = 32

FLOWS = [
    FLOW_AK := FlowDef(1, "AEFJK", "tab:blue"),
    FLOW_BL := FlowDef(2, "BEFJL", "tab:orange"),
    FLOW_CI := FlowDef(3, "CEFI", "tab:green"),
    FLOW_DH := FlowDef(4, "DEFH", "tab:red"),
    FLOW_GM := FlowDef(5, "GFJM", "tab:purple"),
]
N_FLOWS = len(FLOWS)
for i, flow in enumerate(FLOWS):
    flow.idx = i

type Scenario = tuple[str, list[FlowDef]]
SCENARIOS: list[Scenario] = [
    ("AK", [FLOW_AK]),
    ("BL", [FLOW_BL]),
    ("CI", [FLOW_CI]),
    ("DH", [FLOW_DH]),
    ("GM", [FLOW_GM]),
    ("1) AK+CI", [FLOW_AK, FLOW_CI]),
    ("2) AK+BL", [FLOW_AK, FLOW_BL]),
    ("3) AK+CI+DH", [FLOW_AK, FLOW_CI, FLOW_DH]),
    ("4) AK+BL+CI+DH+GM", [FLOW_AK, FLOW_BL, FLOW_CI, FLOW_DH, FLOW_GM]),
]

STRATEGIES: dict[str, MuxSchemeInput] = {
    "Statistical": MuxSchemeStatistical(select_swap_qubit="random", coordinated_decisions=True),
    "Buffer-Space": MuxSchemeBufferSpace(select_swap_qubit="random"),
}


def _mv_for_flow(flow: str, route: list[str], active_flows: set[str]) -> MultiplexingVector:
    """
    Build a MultiplexingVector for one flow under Buffer-Space multiplexing,
    applying the per-link qubit allocations.
    For Statistical mux, this is ignored.
    """
    mv: MultiplexingVector = []
    for u, v in itertools.pairwise(route):
        pair = f"{u}{v}"
        # Default: full TX/RX on uncontested links
        tx_rx = (RX_QUBITS, TX_QUBITS) if pair == "GF" else (TX_QUBITS, RX_QUBITS)

        # --- Buffer-space splits on contested links ---
        if pair == "EF":  # EF contested:
            if active_flows == {"AK", "CI"}:
                # AK+CI: 2 flows on EF -> 16@E, 25@F each
                tx_rx = 16, 25
            elif active_flows == {"AK", "BL"}:
                # AK+BL: 2 flows on EF (and also FJ elsewhere) -> 16@E, 25@F each
                tx_rx = 16, 25
            elif active_flows == {"AK", "CI", "DH"}:
                # AK+CI+DH: 3 flows on EF
                # E: 11 (AK), 11 (CI), 10 (DH)
                # F: 17 (AK), 17 (CI), 16 (DH)
                if flow == "AK":
                    tx_rx = 11, 17
                elif flow == "CI":
                    tx_rx = 11, 17
                elif flow == "DH":
                    tx_rx = 10, 16
            elif active_flows == {"AK", "BL", "CI", "DH", "GM"}:
                # All five: EF has 4 flows (A,B,C,D)
                # E: 8 (AK,BL,CI,DH)
                # F: 17 (AK,BL), 16 (CI,DH)
                if flow in {"AK", "BL"}:
                    tx_rx = 8, 13
                elif flow in {"CI", "DH"}:
                    tx_rx = 8, 12
            else:
                tx_rx = (RX_QUBITS, TX_QUBITS)
        elif pair == "FJ":  # FJ contested:
            if active_flows == {"AK", "BL"}:
                # AK+BL: 2 flows on FJ -> 16@J, 25@F each
                tx_rx = 25, 16  # (F side, J side)
            elif active_flows == {"AK", "BL", "CI", "DH", "GM"}:
                # All five: FJ has 3 flows (AK,BL,GM)
                # F: 17 (AK,BL), 16 (GM); J: 11 (AK,BL), 10 (GM)
                if flow in {"AK", "BL"}:
                    tx_rx = 17, 11
                elif flow == "GM":
                    tx_rx = 16, 10
        mv.extend(tx_rx)

    # print(f"flow: {flow} , route={route}, mv: {mv}")
    return mv


def build_network(mux: MuxSchemeInput, active_flows: Sequence[FlowDef], active_flows_set: set[str]) -> QuantumNetwork:
    b = NetworkBuilder()
    define_topo(b, ch_capacity=(TX_QUBITS, RX_QUBITS))

    b.proactive_centralized(mux=mux)

    if mux_scheme_is_buffer_space(mux):
        # Explicit static paths with per-hop MVs
        for flow in active_flows:
            b.request(
                RoutingPath.static(
                    flow.route, req_id=flow.req_id, bufferspace_mv=_mv_for_flow(flow.label, flow.route, active_flows_set)
                )
            )
    else:
        # Statistical: best-effort usage; no pre-split
        for flow in active_flows:
            b.request(RoutingPath.static(flow.route, req_id=flow.req_id))

    return b.make_network()


def run_simulation(seed: int, args: Args, mux: MuxSchemeInput, active_flows: list[FlowDef]):
    rng.reseed(seed)

    active_flows_set = set(f.label for f in active_flows)
    net = build_network(mux, active_flows, active_flows_set)

    s = Simulator(0, args.sim_duration + CTRL_DELAY, accuracy=1000000, install_to=(log, net))
    s.run()

    # Collect per-source stats in fixed order [AK, BL, CI, DH, GM]
    def _get_rate_fid(flow: FlowDef):
        consume_cnt = RequestCounters.of(net, flow.req_id, (flow.route[0], flow.route[-1]))
        return consume_cnt.get_rate(args.sim_duration), consume_cnt.consumed_avg_fidelity

    stats: list[tuple[float, float]] = []  # [(AK), (BL), (CI), (DH), (GM)] # disabled flows have zero stats
    for flow in FLOWS:
        if flow.label in active_flows_set:
            stats.append(_get_rate_fid(flow))
        else:
            stats.append((0, 0))

    total_decoh = sum(node.get_app(LinkLayer).cnt.n_decoh for node in net.nodes)
    total_swap_conflict = sum(node.get_app(ProactiveForwarder).cnt.n_su_lower[4] for node in net.nodes)

    return stats, total_decoh, total_swap_conflict


class FlowStats(NamedTuple):
    rate_mean: float
    rate_std: float
    fid_mean: float
    fid_std: float


def run_row(args: Args, strategy: str, scenario: Scenario) -> list[FlowStats]:
    mux = STRATEGIES[strategy]
    label, flows = scenario

    flow_rates = [[] for _ in range(N_FLOWS)]
    flow_fids = [[] for _ in range(N_FLOWS)]

    for seed in seed_seq_env(args.runs, 100):
        flow_stats, total_decoh, total_swap_conflict = run_simulation(seed, args, mux, flows)
        print(f"{strategy}, {label}, seed={seed}, decoh={total_decoh}, swap-conflict={total_swap_conflict}")
        for idx, (rate, fid) in enumerate(flow_stats):
            flow_rates[idx].append(rate)
            flow_fids[idx].append(fid)

    return [
        FlowStats(np.mean(rates).item(), np.std(rates).item(), np.mean(fids).item(), np.std(fids).item())
        for rates, fids in zip(flow_rates, flow_fids, strict=True)
    ]


def plot(results: dict[str, list[list[FlowStats]]], args: Args):
    # ==============================
    # Stacked aggregate-rate bars per strategy
    # ==============================

    SCENARIO_FLOWS = {i: tuple(SCENARIOS[i][1]) for i in range(len(SCENARIOS))}

    def find_alone_idx(flow: FlowDef) -> int | None:
        """Find the scenario index where this flow is alone (its baseline)."""
        for i, flows in SCENARIO_FLOWS.items():
            if len(flows) == 1 and flows[0] == flow:
                return i
        return None

    # Find scenario indices for "1) .. 4)" by label
    scenario_1_to_4: dict[int, int] = {}
    for i, (label, _) in enumerate(SCENARIOS):
        if label.startswith("1)"):
            scenario_1_to_4[1] = i
        if label.startswith("2)"):
            scenario_1_to_4[2] = i
        if label.startswith("3)"):
            scenario_1_to_4[3] = i
        if label.startswith("4)"):
            scenario_1_to_4[4] = i

    def stacked_data_for_strategy(res_for_strategy: list[list[FlowStats]]):
        """
        Build a (5 bars) x (5 flows) matrix of contributions:
        bar 0: uncontested SUM (each column = flow baseline when run alone)
        bars 1..4: scenarios 1..4 (each column = that flow's mean rate in the scenario; 0 if absent)
        """
        # Bar labels
        bar_labels = ["Baseline", "AK+CI", "AK+BL", "AK+CI+DH", "AK+BL+CI+DH+GM"]

        # Initialize contributions: rows=bars, cols=flows
        contrib = np.zeros((5, 5), dtype=float)

        # Bar 0: uncontested baselines from 'alone' runs
        for flow in FLOWS:
            i_alone = find_alone_idx(flow)
            if i_alone is not None and res_for_strategy[i_alone][flow.idx]:
                contrib[0, flow.idx] = res_for_strategy[i_alone][flow.idx].rate_mean

        # Bars 1..4: scenarios 1..4
        for k in [1, 2, 3, 4]:
            s_idx = scenario_1_to_4.get(k, None)
            if s_idx is None:
                continue
            for flow in FLOWS:
                if res_for_strategy[s_idx][flow.idx]:
                    contrib[k, flow.idx] = res_for_strategy[s_idx][flow.idx].rate_mean

        return bar_labels, contrib

    def plot_stacked_aggregate_bars(results: dict, strategy_name: str, title: str):
        res = results[strategy_name]
        bar_labels, contrib = stacked_data_for_strategy(res)

        # Plot
        mpl.rcParams.update(
            {
                "font.size": 13,
                "axes.titlesize": 13,
                "axes.labelsize": 13,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "figure.titlesize": 16,
            }
        )
        fig, ax = plt.subplots(figsize=(8.0, 5.2))

        x = np.arange(len(bar_labels))
        bottoms = np.zeros(len(bar_labels), dtype=float)

        # Stack each flow's contribution with a consistent color
        for flow in FLOWS:
            vals = contrib[:, flow.idx]
            ax.bar(x, vals, bottom=bottoms, width=0.65, label=flow.label, color=flow.color)
            bottoms += vals

        ax.set_xticks(x, bar_labels)
        ax.set_xticklabels(bar_labels, rotation=30, ha="right")
        ax.set_ylabel("E2E rate (eps)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

        # Legend: vertical on the right
        fig.subplots_adjust(right=0.80)
        ax.legend(
            title="Flow",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncols=1,
            frameon=False,
            borderaxespad=0.0,
        )

        fig.tight_layout(rect=(0, 0, 0.80, 1))
        return fig

    # Make one stacked chart per strategy
    fig_stack_stat = plot_stacked_aggregate_bars(results, "Statistical", "Throughput for Statistical Multiplexing (eps)")
    fig_stack_buff = plot_stacked_aggregate_bars(results, "Buffer-Space", "Throughput for Buffer-Space  Multiplexing (eps)")

    plt_save((fig_stack_stat, args.plt_stat), (fig_stack_buff, args.plt_buff))


if __name__ == "__main__":
    freeze_support()
    args = Args().parse_args()

    with Pool(processes=args.workers) as pool:
        rows = pool.starmap(run_row, itertools.product([args], STRATEGIES, SCENARIOS))

    results: dict[str, list[list[FlowStats]]] = {strategy: [] for strategy in STRATEGIES}  # strategy->scenario->flow_idx
    for (strategy, scenario_idx), row in zip(itertools.product(STRATEGIES, SCENARIOS), rows, strict=True):
        results[strategy].append(row)

    if args.json:
        with open(args.json, "w") as file:
            json.dump(results, file)

    plot(results, args)
