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

swapping_config = "swap_2_l2r"


def generate_topology(
    nodes: list[str], mem_capacities: list[int], channel_lengths: list[float], capacities: list[tuple[int, int]]
) -> dict:
    """
    Generate a linear topology with explicit memory and channel configurations.

    Args:
        nodes (list[str]): List of node names.
        mem_capacities (list[int]): Number of qubits per node.
        channel_lengths (list[float]): Lengths of quantum channels between adjacent nodes.
        capacities (list[tuple[int, int]]): (left, right) qubit allocation per qchannel.

    Returns:
        dict: A topology dictionary.
    """
    if len(nodes) != len(mem_capacities):
        raise ValueError("mem_capacities must match number of nodes")
    if len(channel_lengths) != len(nodes) - 1:
        raise ValueError("channel_lengths must be len(nodes) - 1")
    if len(capacities) != len(nodes) - 1:
        raise ValueError("capacities must be len(nodes) - 1")

    # Create QNodes
    qnodes = []
    for i, name in enumerate(nodes):
        qnodes.append(
            {
                "name": name,
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": mem_capacities[i],
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
                    ProactiveForwarder(ps=p_swap),
                ],
            }
        )

    # Create Quantum Channels
    qchannels = []
    for i in range(len(nodes) - 1):
        node1, node2 = nodes[i], nodes[i + 1]
        length = channel_lengths[i]
        cap1, cap2 = capacities[i]

        qchannels.append(
            {
                "node1": node1,
                "node2": node2,
                "capacity1": cap1,
                "capacity2": cap2,
                "parameters": {"length": length, "delay": length / light_speed},
            }
        )

    # Classical Channels
    cchannels = []
    for i in range(len(nodes) - 1):
        node1, node2 = nodes[i], nodes[i + 1]
        length = channel_lengths[i]
        cchannels.append({"node1": node1, "node2": node2, "parameters": {"length": length, "delay": length / light_speed}})

    # Controller and links to all nodes
    controller = {"name": "ctrl", "apps": [ProactiveRoutingControllerApp(routing_type="SRSP", swapping=swapping_config)]}
    for node in nodes:
        cchannels.append({"node1": "ctrl", "node2": node, "parameters": {"length": 1.0, "delay": 1.0 / light_speed}})

    return {"qnodes": qnodes, "qchannels": qchannels, "cchannels": cchannels, "controller": controller}


nodes = ["S", "R1", "R2", "D"]
mem_capacities = [4, 4, 4, 4]  # number of qubits per node should be enough for qchannels
channel_lengths = [32, 18, 10]
capacities = [(4, 3), (1, 2), (2, 4)]

json_topology = generate_topology(nodes, mem_capacities, channel_lengths, capacities)
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

e2e_rate = net.get_node("S").get_apps(ProactiveForwarder)[0].e2e_count / sim_duration

print(f"E2E etg rate: {e2e_rate}")
print(f"Expired memories: {total_decohered / total_etg if total_etg > 0 else 0}")
