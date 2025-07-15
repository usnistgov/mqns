import logging

from qns.network import QuantumNetwork, TimingModeEnum
from qns.network.protocol.link_layer import LinkLayer
from qns.network.protocol.proactive_forwarder import ProactiveForwarder
from qns.network.protocol.proactive_routing_controller import ProactiveRoutingControllerApp
from qns.network.route.dijkstra import DijkstraRouteAlgorithm
from qns.network.topology.customtopo import CustomTopology
from qns.simulator.simulator import Simulator
from qns.utils import log
from qns.utils.rnd import set_seed

log.logger.setLevel(logging.DEBUG)

SEED_BASE = 100

light_speed = 2 * 10**5  # km/s

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

swapping_order = "swap_3_asap"  # TODO: here we know there are 3 swaps on all paths

# Multipath settings
routing_type = "MRSP_DYNAMIC"  # Controller installs one path for each S-D request, without qubit-path allocation

# NOTE: Statistical works only with SWAP-ASAP
statistical_mux = True  # enable statistical mux (True) or dynamic EPR affectation (False)

# Quantum channel lengths
ch_S1_R1 = 10
ch_R1_R2 = 10
ch_R2_R3 = 10
ch_R3_D1 = 10
ch_S2_R4 = 10
ch_R4_R2 = 15
ch_R3_D2 = 15


def generate_topology() -> dict:
    """
    Defines the topology with globally declared simulation parameters.

    Returns:
        dict: the topology definition to be used to build the quantum network.
    """
    return {
        "qnodes": [
            {
                "name": "S1",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux),
                ],
            },
            {
                "name": "S2",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux),
                ],
            },
            {
                "name": "D1",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux),
                ],
            },
            {
                "name": "D2",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 1,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux),
                ],
            },
            {
                "name": "R1",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 2,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux),
                ],
            },
            {
                "name": "R4",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 2,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux),
                ],
            },
            {
                "name": "R2",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 3,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux),
                ],
            },
            {
                "name": "R3",
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": 3,
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
                    ProactiveForwarder(ps=p_swap, statistical_mux=statistical_mux),
                ],
            },
        ],
        "qchannels": [
            {"node1": "S1", "node2": "R1", "capacity": 1, "parameters": {"length": ch_S1_R1, "delay": ch_S1_R1 / light_speed}},
            {"node1": "R1", "node2": "R2", "capacity": 1, "parameters": {"length": ch_R1_R2, "delay": ch_R1_R2 / light_speed}},
            {"node1": "R2", "node2": "R3", "capacity": 1, "parameters": {"length": ch_R2_R3, "delay": ch_R2_R3 / light_speed}},
            {"node1": "R3", "node2": "D1", "capacity": 1, "parameters": {"length": ch_R3_D1, "delay": ch_R3_D1 / light_speed}},
            {"node1": "S2", "node2": "R4", "capacity": 1, "parameters": {"length": ch_S2_R4, "delay": ch_S2_R4 / light_speed}},
            {"node1": "R4", "node2": "R2", "capacity": 1, "parameters": {"length": ch_R4_R2, "delay": ch_R4_R2 / light_speed}},
            {"node1": "R3", "node2": "D2", "capacity": 1, "parameters": {"length": ch_R3_D2, "delay": ch_R3_D2 / light_speed}},
        ],
        "cchannels": [
            {"node1": "S1", "node2": "R1", "parameters": {"length": ch_S1_R1, "delay": ch_S1_R1 / light_speed}},
            {"node1": "R1", "node2": "R2", "parameters": {"length": ch_R1_R2, "delay": ch_R1_R2 / light_speed}},
            {"node1": "R2", "node2": "R3", "parameters": {"length": ch_R2_R3, "delay": ch_R2_R3 / light_speed}},
            {"node1": "R3", "node2": "D1", "parameters": {"length": ch_R3_D1, "delay": ch_R3_D1 / light_speed}},
            {"node1": "S2", "node2": "R4", "parameters": {"length": ch_S2_R4, "delay": ch_S2_R4 / light_speed}},
            {"node1": "R4", "node2": "R2", "parameters": {"length": ch_R4_R2, "delay": ch_R4_R2 / light_speed}},
            {"node1": "R3", "node2": "D2", "parameters": {"length": ch_R3_D2, "delay": ch_R3_D2 / light_speed}},
            {"node1": "ctrl", "node2": "S1", "parameters": {"length": 1.0, "delay": 1 / light_speed}},
            {"node1": "ctrl", "node2": "S2", "parameters": {"length": 1.0, "delay": 1 / light_speed}},
            {"node1": "ctrl", "node2": "R1", "parameters": {"length": 1.0, "delay": 1 / light_speed}},
            {"node1": "ctrl", "node2": "R2", "parameters": {"length": 1.0, "delay": 1 / light_speed}},
            {"node1": "ctrl", "node2": "R3", "parameters": {"length": 1.0, "delay": 1 / light_speed}},
            {"node1": "ctrl", "node2": "R4", "parameters": {"length": 1.0, "delay": 1 / light_speed}},
            {"node1": "ctrl", "node2": "D1", "parameters": {"length": 1.0, "delay": 1 / light_speed}},
            {"node1": "ctrl", "node2": "D2", "parameters": {"length": 1.0, "delay": 1 / light_speed}},
        ],
        "controller": {
            "name": "ctrl",
            "apps": [ProactiveRoutingControllerApp(swapping=swapping_order, routing_type=routing_type)],
        },
    }


json_topology = generate_topology()
print(json_topology)

set_seed(SEED_BASE)
s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
log.install(s)

topo = CustomTopology(json_topology)
net = QuantumNetwork(topo=topo, route=DijkstraRouteAlgorithm(), timing_mode=TimingModeEnum.ASYNC)
net.install(s)

s.run()

#### get stats
total_etg = 0
total_decohered = 0
for node in net.get_nodes():
    ll_app = node.get_apps(LinkLayer)[0]
    total_etg += ll_app.etg_count
    total_decohered += ll_app.decoh_count

e2e_rate_1 = net.get_node("S1").get_apps(ProactiveForwarder)[0].e2e_count / sim_duration
e2e_rate_2 = net.get_node("S2").get_apps(ProactiveForwarder)[0].e2e_count / sim_duration

print(f"E2E etg rate [S1-D1]: {e2e_rate_1}")
print(f"E2E etg rate [S2-D2]: {e2e_rate_2}")
print(f"Expired memories: {total_decohered / total_etg if total_etg > 0 else 0}")
