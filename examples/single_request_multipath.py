from mqns.network.network import QuantumNetwork
from mqns.network.proactive import LinkLayer, ProactiveForwarder, ProactiveRoutingController, RoutingPathMulti
from mqns.network.route import YenRouteAlgorithm
from mqns.network.topology import CustomTopology, Topology
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from examples_common.stats import gather_etg_decoh

log.set_default_level("DEBUG")

SEED_BASE = 100

# parameters
sim_duration = 3

fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)

init_fidelity = 0.99
p_swap = 0.5
t_cohere = 0.01  # sec

node_capacity = 4

swapping_policy = "r2l"

# Quantum channel lengths
ch_S_R1 = 10
ch_R1_R2 = 10
ch_R2_R3 = 10
ch_R3_R4 = 10
ch_R4_D = 10
ch_S_R5 = 15
ch_R5_R3 = 15


def build_topology() -> Topology:
    """
    Defines the topology with globally declared simulation parameters.
    """

    return CustomTopology(
        {
            "qnodes": [
                {"name": "S"},
                {"name": "R1"},
                {"name": "R2"},
                {"name": "R3"},
                {"name": "R4"},
                {"name": "R5"},
                {"name": "D"},
            ],
            "qchannels": [
                {
                    "node1": "S",
                    "node2": "R1",
                    "capacity1": 2,
                    "capacity2": 2,
                    "parameters": {"length": ch_S_R1},
                },
                {
                    "node1": "R1",
                    "node2": "R2",
                    "capacity1": 2,
                    "capacity2": 2,
                    "parameters": {"length": ch_R1_R2},
                },
                {
                    "node1": "R2",
                    "node2": "R3",
                    "capacity1": 2,
                    "capacity2": 1,
                    "parameters": {"length": ch_R2_R3},
                },
                {
                    "node1": "R3",
                    "node2": "R4",
                    "capacity1": 2,
                    "capacity2": 2,
                    "parameters": {"length": ch_R3_R4},
                },
                {
                    "node1": "R4",
                    "node2": "D",
                    "capacity1": 2,
                    "capacity2": 4,
                    "parameters": {"length": ch_R4_D},
                },
                {
                    "node1": "S",
                    "node2": "R5",
                    "capacity1": 2,
                    "capacity2": 2,
                    "parameters": {"length": ch_S_R5},
                },
                {
                    "node1": "R5",
                    "node2": "R3",
                    "capacity1": 2,
                    "capacity2": 1,
                    "parameters": {"length": ch_R5_R3},
                },
            ],
            "cchannels": [
                {"node1": "S", "node2": "R1", "parameters": {"length": ch_S_R1}},
                {"node1": "R1", "node2": "R2", "parameters": {"length": ch_R1_R2}},
                {"node1": "R2", "node2": "R3", "parameters": {"length": ch_R2_R3}},
                {"node1": "R3", "node2": "R4", "parameters": {"length": ch_R3_R4}},
                {"node1": "R4", "node2": "D", "parameters": {"length": ch_R4_D}},
                {"node1": "S", "node2": "R5", "parameters": {"length": ch_S_R5}},
                {"node1": "R5", "node2": "R3", "parameters": {"length": ch_R5_R3}},
                {"node1": "ctrl", "node2": "S", "parameters": {"length": 1.0}},
                {"node1": "ctrl", "node2": "R1", "parameters": {"length": 1.0}},
                {"node1": "ctrl", "node2": "R2", "parameters": {"length": 1.0}},
                {"node1": "ctrl", "node2": "R3", "parameters": {"length": 1.0}},
                {"node1": "ctrl", "node2": "R4", "parameters": {"length": 1.0}},
                {"node1": "ctrl", "node2": "R5", "parameters": {"length": 1.0}},
                {"node1": "ctrl", "node2": "D", "parameters": {"length": 1.0}},
            ],
            "controller": {
                "name": "ctrl",
                "apps": [ProactiveRoutingController(RoutingPathMulti("S", "D", swap=swapping_policy))],
            },
        },
        nodes_apps=[
            LinkLayer(
                attempt_rate=entg_attempt_rate,
                init_fidelity=init_fidelity,
                alpha_db_per_km=fiber_alpha,
                eta_d=eta_d,
                eta_s=eta_s,
                frequency=frequency,
            ),
            ProactiveForwarder(ps=p_swap),
        ],
        memory_args={
            "t_cohere": t_cohere,
            "capacity": node_capacity,
        },
    )


set_seed(SEED_BASE)

topo = build_topology()
net = QuantumNetwork(
    topo,
    route=YenRouteAlgorithm(),  # Yen's algo is set here!
)

s = Simulator(0, sim_duration + 5e-06, accuracy=1000000, install_to=(log, net))
s.run()

#### get stats
_, _, decoh_ratio = gather_etg_decoh(net)
e2e_rate = net.get_node("S").get_app(ProactiveForwarder).cnt.n_consumed / sim_duration

print(f"E2E etg rate: {e2e_rate}")
print(f"Expired memories: {decoh_ratio}")
