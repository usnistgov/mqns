from mqns.entity.qchannel import LinkArch, LinkArchDimBkSeq
from mqns.network.proactive import LinkLayer, ProactiveForwarder, ProactiveRoutingController, RoutingPathSingle
from mqns.network.topology.customtopo import CustomTopology, Topo, TopoCChannel, TopoController, TopoQChannel, TopoQNode
from mqns.network.topology.topo import Topology


def _split_channel_capacity(item: int | tuple[int, int]) -> tuple[int, int]:
    return (item, item) if isinstance(item, int) else item


def build_topology(
    *,
    nodes: int | list[str],
    mem_capacity: int | list[int] | None = None,
    t_coherence: float,
    channel_length: float | list[float],
    channel_capacity: int | list[int] | list[tuple[int, int]] | list[int | tuple[int, int]] = 1,
    link_arch: LinkArch | list[LinkArch] = LinkArchDimBkSeq(),
    entg_attempt_rate: float = 50e6,  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)
    init_fidelity: float = 0.99,
    fiber_alpha: float = 0.2,
    eta_d: float = 0.95,
    eta_s: float = 0.95,
    frequency: float = 1e6,  # memory frequency
    p_swap: float = 0.5,
    swap: list[int] | str,
) -> Topology:
    """
    Build a linear topology consisting of zero or more repeaters.

    Args:
        nodes: Number of nodes or list of node names.
        # QuantumMemory
        mem_capacity: Number of memory qubits per node.
        t_coherence: Memory coherence time in seconds.
        # QuantumChannel
        channel_length: Lengths of qchannels between adjacent nodes.
        channel_capacity: (left, right) qubit allocation per qchannel.
        link_arch: Link architecture per qchannel.
        # LinkLayer
        entg_attempt_rate: maximum entanglement attempts per second.
        init_fidelity: fidelity of generated entangled pairs.
        fiber_alpha: fiber loss in dB/km.
        eta_d: detector efficiency.
        eta_s: source efficiency.
        frequency: entanglement source frequency.
        # ProactiveForwarder
        p_swap: probability of successful entanglement swapping.
        # ProactiveRoutingController
        swap: predefined or explicitly specified swapping order.
    """
    if not isinstance(nodes, list):
        assert nodes >= 2, "at least two nodes"
        nodes = [f"R{i}" for i in range(nodes)]
        nodes[0] = "S"
        nodes[-1] = "D"

    n_nodes = len(nodes)
    assert n_nodes >= 2, "at least two nodes"
    n_links = n_nodes - 1

    if isinstance(channel_length, list):
        assert len(channel_length) == n_links, f"channel_length must have {n_links} items"
    else:
        channel_length = [float(channel_length)] * n_links

    if isinstance(channel_capacity, list):
        assert len(channel_capacity) == n_links, f"channel_capacity must have {n_links} items"
    else:
        channel_capacity = [int(channel_capacity)] * n_links

    if isinstance(link_arch, list):
        assert len(link_arch) == n_links, f"link_arch must have {n_links} items"
    else:
        link_arch = [link_arch] * n_links

    if mem_capacity is None:
        mem_capacity = [0]
        for caps in channel_capacity:
            capL, capR = _split_channel_capacity(caps)
            mem_capacity[-1] += capL
            mem_capacity.append(capR)
    elif isinstance(mem_capacity, list):
        assert len(mem_capacity) == len(nodes), f"mem_capacity must have {len(nodes)} items"
    else:
        mem_capacity = [mem_capacity] * n_nodes

    qnodes: list[TopoQNode] = []
    for name, mem_capacity in zip(nodes, mem_capacity):
        qnodes.append(
            {
                "name": name,
                "memory": {
                    "decoherence_rate": 1 / t_coherence,
                    "capacity": mem_capacity,
                },
            }
        )

    qchannels: list[TopoQChannel] = []
    cchannels: list[TopoCChannel] = []
    for i, (length, caps, la) in enumerate(zip(channel_length, channel_capacity, link_arch)):
        node1, node2 = nodes[i], nodes[i + 1]
        cap1, cap2 = _split_channel_capacity(caps)
        qchannels.append(
            {
                "node1": node1,
                "node2": node2,
                "capacity1": cap1,
                "capacity2": cap2,
                "parameters": {"length": length, "link_arch": la},
            }
        )
        cchannels.append({"node1": node1, "node2": node2, "parameters": {"length": length}})

    controller: TopoController = {
        "name": "ctrl",
        "apps": [ProactiveRoutingController(RoutingPathSingle("S", "D", swap=swap))],
    }
    for node in nodes:
        cchannels.append({"node1": "ctrl", "node2": node, "parameters": {"length": 1.0}})

    topo: Topo = {"qnodes": qnodes, "qchannels": qchannels, "cchannels": cchannels, "controller": controller}
    return CustomTopology(
        topo,
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
    )
