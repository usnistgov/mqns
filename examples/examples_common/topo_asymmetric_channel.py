from qns.network.protocol import LinkLayer, ProactiveForwarder, ProactiveRoutingControllerApp
from qns.network.topology.customtopo import CustomTopology, Topo, TopoCChannel, TopoController, TopoQChannel, TopoQNode
from qns.network.topology.topo import Topology

# parameters
fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)
init_fidelity = 0.99
p_swap = 0.5


def build_topology(
    *,
    nodes: list[str],
    mem_capacities: list[int],
    channel_lengths: list[float],
    capacities: list[tuple[int, int]],
    t_coherence: float,
    swapping_order: str,
) -> Topology:
    """
    Generate a linear topology with explicit memory and channel configurations.

    Args:
        nodes (list[str]): List of node names.
        mem_capacities (list[int]): Number of qubits per node.
        channel_lengths (list[float]): Lengths of quantum channels between adjacent nodes.
        capacities (list[tuple[int, int]]): (left, right) qubit allocation per qchannel.
    """
    if len(nodes) != len(mem_capacities):
        raise ValueError("mem_capacities must match number of nodes")
    if len(channel_lengths) != len(nodes) - 1:
        raise ValueError("channel_lengths must be len(nodes) - 1")
    if len(capacities) != len(nodes) - 1:
        raise ValueError("capacities must be len(nodes) - 1")

    qnodes: list[TopoQNode] = []
    for name, mem_capacity in zip(nodes, mem_capacities):
        qnodes.append(
            {
                "name": name,
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": mem_capacity,
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

    qchannels: list[TopoQChannel] = []
    cchannels: list[TopoCChannel] = []
    for i, (length, (cap1, cap2)) in enumerate(zip(channel_lengths, capacities)):
        node1, node2 = nodes[i], nodes[i + 1]
        qchannels.append(
            {
                "node1": node1,
                "node2": node2,
                "capacity1": cap1,
                "capacity2": cap2,
                "parameters": {"length": length},
            }
        )
        cchannels.append({"node1": node1, "node2": node2, "parameters": {"length": length}})

    controller: TopoController = {
        "name": "ctrl",
        "apps": [ProactiveRoutingControllerApp(routing_type="SRSP", swapping=swapping_order)],
    }
    for node in nodes:
        cchannels.append({"node1": "ctrl", "node2": node, "parameters": {"length": 1.0}})

    topo: Topo = {"qnodes": qnodes, "qchannels": qchannels, "cchannels": cchannels, "controller": controller}
    return CustomTopology(topo)
