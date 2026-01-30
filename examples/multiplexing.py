import json

import numpy as np
from tap import Tap

from mqns.network.network import QuantumNetwork
from mqns.network.proactive import (
    LinkLayer,
    MuxScheme,
    MuxSchemeBufferSpace,
    MuxSchemeStatistical,
    ProactiveForwarder,
    ProactiveRoutingController,
    QubitAllocationType,
    RoutingPath,
    RoutingPathStatic,
)
from mqns.network.proactive.message import MultiplexingVector
from mqns.network.topology.customtopo import CustomTopology, TopoCChannel, Topology, TopoQChannel, TopoQNode
from mqns.simulator import Simulator
from mqns.utils import log, rng

from examples_common.plotting import mpl, plt, plt_save


# ------------------------------
# CLI
# ------------------------------
class Args(Tap):
    runs: int = 3
    json: str = ""
    plt_stat: str = ""  # Statistical plot filename
    plt_buff: str = ""  # Buffer-Space plot filename


args = Args().parse_args()
log.set_default_level("CRITICAL")

# ------------------------------
# Paper-like parameters
# ------------------------------
SEED_BASE = 100
CTRL_DELAY = 5e-06
sim_duration = 3.0

fiber_alpha = 0.17  # dB/km
eta_d = 0.5
eta_s = 0.8
frequency = 1e6

init_fidelity = 0.99  # base Bell-pair fidelity
p_swap = 0.5
swapping_policy = "asap"

t_cohere = 100e-3

TX_QUBITS = 50
RX_QUBITS = 32

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
KM20 = 30
QCHANNELS = [
    # left spokes -> E
    ("A", "E"),
    ("B", "E"),
    ("C", "E"),
    ("D", "E"),
    # middle trunks
    ("E", "F"),
    ("F", "J"),
    # right spokes from J
    ("J", "K"),
    ("J", "L"),
    ("J", "M"),
    # bottom spokes from F
    ("G", "F"),
    ("F", "H"),
    ("F", "I"),
]
CCHANNELS = QCHANNELS  # same topology for classical

# ------------------------------
# Flow routes
ROUTE = {
    ("A", "K"): ["A", "E", "F", "J", "K"],
    ("B", "L"): ["B", "E", "F", "J", "L"],
    ("C", "I"): ["C", "E", "F", "I"],
    ("D", "H"): ["D", "E", "F", "H"],
    ("G", "M"): ["G", "F", "J", "M"],
}

SWAP = {
    ("A", "K"): [1, 0, 0, 0, 1],
    ("B", "L"): [1, 0, 0, 0, 1],
    ("C", "I"): [1, 0, 0, 1],
    ("D", "H"): [1, 0, 0, 1],
    ("G", "M"): [1, 0, 0, 1],
}

# Uncontested links use full TX/RX (50/32)
FULL = (TX_QUBITS, RX_QUBITS)
FULL_EF = (RX_QUBITS, TX_QUBITS)

# ------------------------------
# Scenarios
# ------------------------------
SCENARIOS = [
    ("AK", [("A", "K")]),
    ("BL", [("B", "L")]),
    ("CI", [("C", "I")]),
    ("DH", [("D", "H")]),
    ("GM", [("G", "M")]),
    ("1) AK+CI", [("A", "K"), ("C", "I")]),
    ("2) AK+BL", [("A", "K"), ("B", "L")]),
    ("3) AK+CI+DH", [("A", "K"), ("C", "I"), ("D", "H")]),
    ("4) AK+BL+CI+DH+GM", [("A", "K"), ("B", "L"), ("C", "I"), ("D", "H"), ("G", "M")]),
]

# Flow order used by run_simulation return
FLOW_ORDER = [("A", "K"), ("B", "L"), ("C", "I"), ("D", "H"), ("G", "M")]
FLOW_LABELS = ["AK", "BL", "CI", "DH", "GM"]
FLOW_IDX = {f: i for i, f in enumerate(FLOW_ORDER)}


def mv_for_flow(flow: tuple[str, str], active_flows: set[tuple[str, str]]):
    """
    Build a MultiplexingVector for one flow under Buffer-Space multiplexing,
    applying the per-link qubit allocations.
    For Statistical mux, this is ignored.
    """
    route = ROUTE[flow]
    mv: MultiplexingVector = []
    for u, v in zip(route[:-1], route[1:]):
        pair = (u, v)
        # Default: full TX/RX on uncontested links
        tx_rx = FULL_EF if pair == ("G", "F") else FULL

        # --- Buffer-space splits on contested links ---
        # EF contested:
        if pair == ("E", "F"):
            if active_flows == {("A", "K"), ("C", "I")}:
                # AK+CI: 2 flows on EF -> 16@E, 25@F each
                tx_rx = (16, 25)
            elif active_flows == {("A", "K"), ("B", "L")}:
                # AK+BL: 2 flows on EF (and also FJ elsewhere) -> 16@E, 25@F each
                tx_rx = (16, 25)
            elif active_flows == {("A", "K"), ("C", "I"), ("D", "H")}:
                # AK+CI+DH: 3 flows on EF
                # E: 11 (AK), 11 (CI), 10 (DH)
                # F: 17 (AK), 17 (CI), 16 (DH)
                if flow == ("A", "K"):
                    tx_rx = (11, 17)
                elif flow == ("C", "I"):
                    tx_rx = (11, 17)
                elif flow == ("D", "H"):
                    tx_rx = (10, 16)
            elif active_flows == {("A", "K"), ("B", "L"), ("C", "I"), ("D", "H"), ("G", "M")}:
                # All five: EF has 4 flows (A,B,C,D)
                # E: 8 (AK,BL,CI,DH)
                # F: 17 (AK,BL), 16 (CI,DH)
                if flow in {("A", "K"), ("B", "L")}:
                    tx_rx = (8, 13)
                elif flow in {("C", "I"), ("D", "H")}:
                    tx_rx = (8, 12)
            else:
                tx_rx = FULL_EF
        # FJ contested:
        elif pair == ("F", "J"):
            if active_flows == {("A", "K"), ("B", "L")}:
                # AK+BL: 2 flows on FJ -> 16@J, 25@F each
                tx_rx = (25, 16)  # (F side, J side)
            elif active_flows == {("A", "K"), ("B", "L"), ("C", "I"), ("D", "H"), ("G", "M")}:
                # All five: FJ has 3 flows (AK,BL,GM)
                # F: 17 (AK,BL), 16 (GM); J: 11 (AK,BL), 10 (GM)
                if flow in {("A", "K"), ("B", "L")}:
                    tx_rx = (17, 11)
                elif flow == ("G", "M"):
                    tx_rx = (16, 10)
        mv.append(tx_rx)

    # print(f"flow: {flow} , route={route}, mv: {mv}")
    return mv


def build_topology(t_cohere: float, mux: MuxScheme, active_flows: list[tuple[str, str]]) -> Topology:
    # Install paths differ for buffer-space vs statistical
    install_paths: list[RoutingPath] = []
    if isinstance(mux, MuxSchemeBufferSpace):
        # Explicit static paths with per-hop MVs
        active_flows_set = set(active_flows)
        for flow in active_flows:
            install_paths.append(RoutingPathStatic(ROUTE[flow], m_v=mv_for_flow(flow, active_flows_set), swap=swapping_policy))
    else:
        # Statistical: best-effort usage; no pre-split
        for flow in active_flows:
            install_paths.append(RoutingPathStatic(ROUTE[flow], m_v=QubitAllocationType.DISABLED, swap=swapping_policy))

    def _node(name, cap) -> TopoQNode:
        return {"name": name, "memory": {"t_cohere": t_cohere, "capacity": cap}}

    qnodes = (
        [_node(n, TX_QUBITS) for n in "ABCD"]
        + [_node(n, RX_QUBITS) for n in "KLMIHG"]
        + [_node("E", 5 * RX_QUBITS), _node("F", 5 * TX_QUBITS), _node("J", 3 * TX_QUBITS + RX_QUBITS)]
    )

    def qch(n1, n2) -> TopoQChannel:
        if (n1, n2) in [("E", "F"), ("G", "F")]:
            return {"node1": n1, "node2": n2, "capacity1": RX_QUBITS, "capacity2": TX_QUBITS, "parameters": {"length": KM20}}
        else:
            return {"node1": n1, "node2": n2, "capacity1": TX_QUBITS, "capacity2": RX_QUBITS, "parameters": {"length": KM20}}

    qchannels = [qch(a, b) for a, b in QCHANNELS]
    cchannels = [TopoCChannel({"node1": a, "node2": b, "parameters": {"length": KM20}}) for a, b in CCHANNELS] + [
        TopoCChannel({"node1": "ctrl", "node2": n, "parameters": {"delay": CTRL_DELAY}}) for n in "ABCDEFJKLMGHI"
    ]

    return CustomTopology(
        {
            "qnodes": qnodes,
            "qchannels": qchannels,
            "cchannels": cchannels,
            "controller": {"name": "ctrl", "apps": [ProactiveRoutingController(install_paths)]},
        },
        nodes_apps=[
            LinkLayer(
                init_fidelity=init_fidelity,
                alpha_db_per_km=fiber_alpha,
                eta_d=eta_d,
                eta_s=eta_s,
                frequency=frequency,
            ),
            ProactiveForwarder(
                ps=p_swap,
                mux=mux,
            ),
        ],
    )


def run_simulation(t_cohere: float, mux: MuxScheme, seed: int, active_flows: list[tuple[str, str]]):
    rng.reseed(seed)

    topo = build_topology(t_cohere, mux, active_flows)
    net = QuantumNetwork(topo)

    s = Simulator(0, sim_duration + CTRL_DELAY, accuracy=1000000, install_to=(log, net))
    s.run()

    # Collect per-source stats in fixed order [AK, BL, CI, DH, GM]
    def _get_rate_fid(src):
        node = net.get_node(src)
        fw = node.get_app(ProactiveForwarder)
        return (fw.cnt.n_consumed / sim_duration, fw.cnt.consumed_avg_fidelity)

    stats: list[tuple[float, float]] = []  # [(AK), (BL), (CI), (DH), (GM)] # disabled flows have zero stats
    for flow in FLOW_ORDER:
        if flow in active_flows:
            stats.append(_get_rate_fid(flow[0]))
        else:
            stats.append((0, 0))

    total_decoh = sum((node.get_app(LinkLayer).cnt.n_decoh for node in net.nodes))
    total_swap_conflict = sum((node.get_app(ProactiveForwarder).cnt.n_swap_conflict for node in net.nodes))

    return stats, total_decoh, total_swap_conflict


# ------------------------------
# Strategies
STRATEGIES: dict[str, MuxScheme] = {
    "Statistical": MuxSchemeStatistical(
        select_swap_qubit=MuxSchemeStatistical.SelectSwapQubit_random,
        coordinated_decisions=True,
    ),
    "Buffer-Space": MuxSchemeBufferSpace(
        select_swap_qubit=MuxSchemeBufferSpace.SelectSwapQubit_random,
    ),
}

# results[strategy][scenario_idx][flow_idx]
results = {name: {i: {j: [] for j in range(len(FLOW_ORDER))} for i in range(len(SCENARIOS))} for name in STRATEGIES}

for s_idx, (label, flows) in enumerate(SCENARIOS):
    for strategy, mux in STRATEGIES.items():
        flow_rates = [[] for _ in range(len(FLOW_ORDER))]
        flow_fids = [[] for _ in range(len(FLOW_ORDER))]
        for i in range(args.runs):
            flow_stats, total_decoh, total_swap_conflict = run_simulation(t_cohere, mux, SEED_BASE + i, flows)
            print(f"{strategy}, {label}, run #{i}, decoh={total_decoh}, swap-conflict={total_swap_conflict}")
            for idx, (rate, fid) in enumerate(flow_stats):
                flow_rates[idx].append(rate)
                flow_fids[idx].append(fid)
        for idx in range(len(FLOW_ORDER)):
            mean_rate = np.mean(flow_rates[idx])
            std_rate = np.std(flow_rates[idx])
            mean_fid = np.mean(flow_fids[idx])
            std_fid = np.std(flow_fids[idx])
            results[strategy][s_idx][idx].append((mean_rate, std_rate, mean_fid, std_fid))

# Optional JSON dump
if args.json:
    with open(args.json, "w") as f:
        json.dump(results, f)


# ==============================
# Stacked aggregate-rate bars per strategy
# ==============================

# Flow colors
FLOW_COLORS = {
    "AK": "tab:blue",
    "BL": "tab:orange",
    "CI": "tab:green",
    "DH": "tab:red",
    "GM": "tab:purple",
}

SCENARIO_FLOWS = {i: tuple(SCENARIOS[i][1]) for i in range(len(SCENARIOS))}


def find_alone_idx(flow: tuple[str, str]) -> int | None:
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


def stacked_data_for_strategy(res_for_strategy: dict):
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
    for col, flow in enumerate(FLOW_ORDER):
        i_alone = find_alone_idx(flow)
        if i_alone is not None and res_for_strategy[i_alone][FLOW_IDX[flow]]:
            contrib[0, col] = res_for_strategy[i_alone][FLOW_IDX[flow]][0][0]  # mean_rate

    # Bars 1..4: scenarios 1..4
    for k in [1, 2, 3, 4]:
        s_idx = scenario_1_to_4.get(k, None)
        if s_idx is None:
            continue
        for col, flow in enumerate(FLOW_ORDER):
            # results[strategy][s_idx][flow_idx][0] -> (mean_rate, std_rate, mean_fid, std_fid)
            if res_for_strategy[s_idx][FLOW_IDX[flow]]:
                contrib[k, col] = res_for_strategy[s_idx][FLOW_IDX[flow]][0][0]

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
    for col, label in enumerate(FLOW_LABELS):
        vals = contrib[:, col]
        ax.bar(x, vals, bottom=bottoms, width=0.65, label=label, color=FLOW_COLORS[label])
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
