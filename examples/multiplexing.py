import json
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from tap import Tap

from mqns.network.builder import CTRL_DELAY, NetworkBuilder
from mqns.network.proactive import (
    MultiplexingVector,
    MuxScheme,
    MuxSchemeBufferSpace,
    MuxSchemeStatistical,
    ProactiveForwarder,
    QubitAllocationType,
    RoutingPathStatic,
)
from mqns.network.protocol.link_layer import LinkLayer
from mqns.simulator import Simulator
from mqns.utils import log, rng

from examples_common.plotting import mpl, plt, plt_save

log.set_default_level("CRITICAL")


class Args(Tap):
    runs: int = 3  # number of trials per parameter set
    sim_duration: float = 3.0  # simulation duration in seconds
    json: str = ""  # save results as JSON file
    plt_stat: str = ""  # save Statistical plot as image file
    plt_buff: str = ""  # save Buffer-Space plot as image file


SEED_BASE = 100
TX_QUBITS = 50
RX_QUBITS = 32


class FlowDef:
    def __init__(self, route: Sequence[str], color: str):
        self.route = list(route)
        self.label = f"{self.route[0]}{self.route[-1]}"
        self.color = color
        self.idx = -1


FLOWS = [
    FLOW_AK := FlowDef("AEFJK", "tab:blue"),
    FLOW_BL := FlowDef("BEFJL", "tab:orange"),
    FLOW_CI := FlowDef("CEFI", "tab:green"),
    FLOW_DH := FlowDef("DEFH", "tab:red"),
    FLOW_GM := FlowDef("GFJM", "tab:purple"),
]
N_FLOWS = len(FLOWS)
for i, flow in enumerate(FLOWS):
    flow.idx = i

SCENARIOS: list[tuple[str, list[FlowDef]]] = [
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

STRATEGIES: dict[str, MuxScheme] = {
    "Statistical": MuxSchemeStatistical(
        select_swap_qubit=MuxSchemeStatistical.SelectSwapQubit_random,
        coordinated_decisions=True,
    ),
    "Buffer-Space": MuxSchemeBufferSpace(
        select_swap_qubit=MuxSchemeBufferSpace.SelectSwapQubit_random,
    ),
}


def _mv_for_flow(flow: str, route: list[str], active_flows: set[str]):
    """
    Build a MultiplexingVector for one flow under Buffer-Space multiplexing,
    applying the per-link qubit allocations.
    For Statistical mux, this is ignored.
    """
    mv: MultiplexingVector = []
    for u, v in zip(route[:-1], route[1:]):
        pair = f"{u}{v}"
        # Default: full TX/RX on uncontested links
        tx_rx = (RX_QUBITS, TX_QUBITS) if pair == "GF" else (TX_QUBITS, RX_QUBITS)

        # --- Buffer-space splits on contested links ---
        if pair == "EF":  # EF contested:
            if active_flows == {"AK", "CI"}:
                # AK+CI: 2 flows on EF -> 16@E, 25@F each
                tx_rx = (16, 25)
            elif active_flows == {"AK", "BL"}:
                # AK+BL: 2 flows on EF (and also FJ elsewhere) -> 16@E, 25@F each
                tx_rx = (16, 25)
            elif active_flows == {"AK", "CI", "DH"}:
                # AK+CI+DH: 3 flows on EF
                # E: 11 (AK), 11 (CI), 10 (DH)
                # F: 17 (AK), 17 (CI), 16 (DH)
                if flow == "AK":
                    tx_rx = (11, 17)
                elif flow == "CI":
                    tx_rx = (11, 17)
                elif flow == "DH":
                    tx_rx = (10, 16)
            elif active_flows == {"AK", "BL", "CI", "DH", "GM"}:
                # All five: EF has 4 flows (A,B,C,D)
                # E: 8 (AK,BL,CI,DH)
                # F: 17 (AK,BL), 16 (CI,DH)
                if flow in {"AK", "BL"}:
                    tx_rx = (8, 13)
                elif flow in {"CI", "DH"}:
                    tx_rx = (8, 12)
            else:
                tx_rx = (RX_QUBITS, TX_QUBITS)
        elif pair == "FJ":  # FJ contested:
            if active_flows == {"AK", "BL"}:
                # AK+BL: 2 flows on FJ -> 16@J, 25@F each
                tx_rx = (25, 16)  # (F side, J side)
            elif active_flows == {"AK", "BL", "CI", "DH", "GM"}:
                # All five: FJ has 3 flows (AK,BL,GM)
                # F: 17 (AK,BL), 16 (GM); J: 11 (AK,BL), 10 (GM)
                if flow in {"AK", "BL"}:
                    tx_rx = (17, 11)
                elif flow == "GM":
                    tx_rx = (16, 10)
        mv.append(tx_rx)

    # print(f"flow: {flow} , route={route}, mv: {mv}")
    return mv


def build_network(mux: MuxScheme, active_flows: Sequence[FlowDef]):
    b = NetworkBuilder()
    # ------------------------------
    # 13-node topology (A..M) with shared trunks EF and FJ
    # All quantum links 30 km
    #
    # A                         K
    #  \                       /
    # B-\                     /
    #    +E--------F--------J+--L
    # C-/         /|\         \
    #  /         / | \         \
    # D         G  H  I         M
    # ------------------------------
    b.topo(
        channels=[
            # left spokes -> E
            (("A", "E"), 30, (TX_QUBITS, RX_QUBITS)),
            (("B", "E"), 30, (TX_QUBITS, RX_QUBITS)),
            (("C", "E"), 30, (TX_QUBITS, RX_QUBITS)),
            (("D", "E"), 30, (TX_QUBITS, RX_QUBITS)),
            # middle trunks
            (("E", "F"), 30, (RX_QUBITS, TX_QUBITS)),
            (("F", "J"), 30, (TX_QUBITS, RX_QUBITS)),
            # right spokes from J
            (("J", "K"), 30, (TX_QUBITS, RX_QUBITS)),
            (("J", "L"), 30, (TX_QUBITS, RX_QUBITS)),
            (("J", "M"), 30, (TX_QUBITS, RX_QUBITS)),
            # bottom spokes from F
            (("G", "F"), 30, (RX_QUBITS, TX_QUBITS)),
            (("F", "H"), 30, (TX_QUBITS, RX_QUBITS)),
            (("F", "I"), 30, (TX_QUBITS, RX_QUBITS)),
        ],
        fiber_alpha=0.17,
        eta_d=0.5,
        eta_s=0.8,
        t_cohere=0.1,
    )

    b.proactive_centralized(mux=mux)

    if isinstance(mux, MuxSchemeBufferSpace):
        # Explicit static paths with per-hop MVs
        active_flows_set = set(f.label for f in active_flows)
        for flow in active_flows:
            b.path(RoutingPathStatic(flow.route, m_v=_mv_for_flow(flow.label, flow.route, active_flows_set), swap="asap"))
    else:
        # Statistical: best-effort usage; no pre-split
        for flow in active_flows:
            b.path(RoutingPathStatic(flow.route, m_v=QubitAllocationType.DISABLED, swap="asap"))

    return b.make_network()


def run_simulation(seed: int, args: Args, mux: MuxScheme, active_flows: list[FlowDef]):
    rng.reseed(seed)

    net = build_network(mux, active_flows)

    s = Simulator(0, args.sim_duration + CTRL_DELAY, accuracy=1000000, install_to=(log, net))
    s.run()

    # Collect per-source stats in fixed order [AK, BL, CI, DH, GM]
    def _get_rate_fid(flow: FlowDef):
        node = net.get_node(flow.route[0])
        fw = node.get_app(ProactiveForwarder)
        return (fw.cnt.n_consumed / args.sim_duration, fw.cnt.consumed_avg_fidelity)

    stats: list[tuple[float, float]] = []  # [(AK), (BL), (CI), (DH), (GM)] # disabled flows have zero stats
    for flow in FLOWS:
        if flow in active_flows:
            stats.append(_get_rate_fid(flow))
        else:
            stats.append((0, 0))

    total_decoh = sum((node.get_app(LinkLayer).cnt.n_decoh for node in net.nodes))
    total_swap_conflict = sum((node.get_app(ProactiveForwarder).cnt.n_swap_conflict for node in net.nodes))

    return stats, total_decoh, total_swap_conflict


class FlowStats(NamedTuple):
    rate_mean: float
    rate_std: float
    fid_mean: float
    fid_std: float


def run_row(args: Args, strategy: str, scenario: int) -> list[FlowStats]:
    mux = STRATEGIES[strategy]
    label, flows = SCENARIOS[scenario]

    flow_rates = [[] for _ in range(N_FLOWS)]
    flow_fids = [[] for _ in range(N_FLOWS)]

    for i in range(args.runs):
        flow_stats, total_decoh, total_swap_conflict = run_simulation(SEED_BASE + i, args, mux, flows)
        print(f"{strategy}, {label}, run #{i}, decoh={total_decoh}, swap-conflict={total_swap_conflict}")
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
    args = Args().parse_args()

    results: dict[str, list[list[FlowStats]]] = {}  # strategy->scenario->flow_idx
    for strategy in STRATEGIES:
        results[strategy] = []
        for scenario in range(len(SCENARIOS)):
            row = run_row(args, strategy, scenario)
            results[strategy].append(row)

    if args.json:
        with open(args.json, "w") as file:
            json.dump(results, file)

    plot(results, args)
