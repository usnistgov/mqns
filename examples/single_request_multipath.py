from qns.network.network import QuantumNetwork
from qns.network.protocol import LinkLayer, ProactiveForwarder, ProactiveRoutingControllerApp
from qns.network.route import YenRouteAlgorithm
from qns.network.topology.customtopo import CustomTopology, Topo, TopoQNode
from qns.simulator import Simulator
from qns.utils import log, set_seed

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
t_coherence = 0.01  # sec

node_capacity = 4

swapping_policy = "r2l"

# Multipath settings
routing_type = "SRMP_STATIC"  # Controller installs multiple paths for a single S-D request, with qubit-path allocation

# NOTE: Non-isolated paths does not work with SWAP-ASAP
isolate_paths = True  # Routers can/cannot swap qubits allocated to different paths (but serving same S-D request)

# Quantum channel lengths
ch_S_R1 = 10
ch_R1_R2 = 10
ch_R2_R3 = 10
ch_R3_R4 = 10
ch_R4_D = 10
ch_S_R5 = 15
ch_R5_R3 = 15


def generate_topology() -> Topo:
    """
    Defines the topology with globally declared simulation parameters.

    Returns:
        dict: the topology definition to be used to build the quantum network.
    """

    # Create QNodes
    qnodes: list[TopoQNode] = []
    for name in ["S", "R1", "R2", "R3", "R4", "R5", "D"]:
        qnodes.append(
            {
                "name": name,
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": node_capacity,
                },
                "apps": [
                    LinkLayer(
                        attempt_rate=entg_attempt_rate,
                        init_fidelity=init_fidelity,
                        alpha_db_per_km=fiber_alpha,
                        eta_d=eta_d,
                        eta_s=eta_s,
                        frequency=frequency,
                    ),
                    ProactiveForwarder(ps=p_swap, isolate_paths=isolate_paths),
                ],
            }
        )

    return {
        "qnodes": qnodes,
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
            "apps": [ProactiveRoutingControllerApp(swapping_policy=swapping_policy, routing_type=routing_type)],
        },
    }


json_topology = generate_topology()
print(json_topology)

set_seed(SEED_BASE)
s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
log.install(s)

topo = CustomTopology(json_topology)
net = QuantumNetwork(
    topo=topo,
    route=YenRouteAlgorithm(),  # Yen's algo is set here!
)
net.install(s)

s.run()

#### get stats
_, _, decoh_ratio = gather_etg_decoh(net)
e2e_rate = net.get_node("S").get_app(ProactiveForwarder).cnt.n_consumed / sim_duration

print(f"E2E etg rate: {e2e_rate}")
print(f"Expired memories: {decoh_ratio}")
