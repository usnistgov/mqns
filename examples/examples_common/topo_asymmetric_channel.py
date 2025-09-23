from typing import cast

from mqns.entity.qchannel import LinkArch, LinkArchDimBkSeq
from mqns.network.proactive import LinkLayer, ProactiveForwarder, ProactiveRoutingController, RoutingPathSingle
from mqns.network.topology.customtopo import CustomTopology, Topo, TopoCChannel, TopoController, TopoQChannel, TopoQNode
from mqns.network.topology.topo import Topology

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
    ch_lengths: list[float],
    ch_capacities: list[tuple[int, int]],
    link_architectures: list[LinkArch] | None = None,
    t_coherence: float,
    swapping_order: str,
) -> Topology:
    """
    Generate a linear topology with explicit memory and channel configurations.

    Args:
        nodes: List of node names.
        mem_capacities: Number of qubits per node.
        ch_lengths: Lengths of quantum channels between adjacent nodes.
        ch_capacities: (left, right) qubit allocation per qchannel.
        link_architectures: Link architecture per qchannel, default is DIM_BK_SEQ.
    """
    if len(mem_capacities) != len(nodes):
        raise ValueError(f"mem_capacities must have {len(nodes)} items")
    n_links = len(nodes) - 1
    if len(ch_lengths) != n_links:
        raise ValueError(f"ch_lengths must have {n_links} items")
    if len(ch_capacities) != n_links:
        raise ValueError(f"ch_capacities must have {n_links} items")
    if link_architectures is None:
        link_architectures = [cast(LinkArch, LinkArchDimBkSeq())] * n_links
    elif len(link_architectures) != n_links:
        raise ValueError(f"link_architectures must have {n_links} items")

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
    for i, (length, (cap1, cap2), link_arch) in enumerate(zip(ch_lengths, ch_capacities, link_architectures)):
        node1, node2 = nodes[i], nodes[i + 1]
        qchannels.append(
            {
                "node1": node1,
                "node2": node2,
                "capacity1": cap1,
                "capacity2": cap2,
                "parameters": {"length": length, "link_arch": link_arch},
            }
        )
        cchannels.append({"node1": node1, "node2": node2, "parameters": {"length": length}})

    controller: TopoController = {
        "name": "ctrl",
        "apps": [ProactiveRoutingController(RoutingPathSingle("S", "D", swap=swapping_order))],
    }
    for node in nodes:
        cchannels.append({"node1": "ctrl", "node2": node, "parameters": {"length": 1.0}})

    topo: Topo = {"qnodes": qnodes, "qchannels": qchannels, "cchannels": cchannels, "controller": controller}
    return CustomTopology(topo)
