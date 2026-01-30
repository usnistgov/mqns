from collections.abc import Sequence

from mqns.entity.qchannel import LinkArch, LinkArchDimBkSeq
from mqns.network.proactive import LinkLayer, ProactiveForwarder, ProactiveRoutingController, RoutingPathSingle
from mqns.network.topology.customtopo import CustomTopology, Topo, TopoCChannel, TopoController, TopoQChannel, TopoQNode
from mqns.network.topology.topo import Topology


def _broadcast[T](name: str, input: T | Sequence[T], n: int) -> Sequence[T]:
    if isinstance(input, Sequence):
        assert len(input) == n, f"{name} must have {n} items"
        return input
    return [input] * n


def _split_channel_capacity(item: int | tuple[int, int]) -> tuple[int, int]:
    return (item, item) if isinstance(item, int) else item


CTRL_DELAY = 5e-06
"""
Delay of the classic channels between the controller and each QNode, in seconds.

In most examples, the overall simulation duration is increased by this value,
so that the QNodes can perform entanglement forwarding for the full intended duration.
"""


def build_topology(
    *,
    nodes: int | Sequence[str],
    mem_capacity: int | Sequence[int] | None = None,
    t_cohere: float = 0.02,
    channel_length: float | Sequence[float],
    channel_capacity: int | Sequence[int | tuple[int, int]] = 1,
    link_arch: LinkArch | Sequence[LinkArch] = LinkArchDimBkSeq(),
    entg_attempt_rate: float = 50e6,
    init_fidelity: float = 0.99,
    fiber_alpha: float = 0.2,
    eta_d: float = 0.95,
    eta_s: float = 0.95,
    frequency: float = 1e6,
    p_swap: float = 0.5,
    swap: list[int] | str,
    swap_cutoff: list[float] | None = None,
) -> Topology:
    """
    Build a linear topology consisting of zero or more repeaters.

    Args:
        nodes: Number of nodes or list of node names.
        # QuantumMemory
        mem_capacity: Number of memory qubits per node.
        t_cohere: Memory coherence time in seconds.
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
        swap_cutoff: cutoff times.
    """
    if isinstance(nodes, int):
        assert nodes >= 2, "at least two nodes"
        nodes = [f"R{i}" for i in range(nodes)]
        nodes[0] = "S"
        nodes[-1] = "D"

    n_nodes = len(nodes)
    assert n_nodes >= 2, "at least two nodes"
    n_links = n_nodes - 1

    channel_length = _broadcast("channel_length", channel_length, n_links)
    channel_capacity = _broadcast("channel_capacity", channel_capacity, n_links)
    link_arch = _broadcast("link_arch", link_arch, n_links)

    if mem_capacity is None:
        mem_capacity = [0]
        for caps in channel_capacity:
            capL, capR = _split_channel_capacity(caps)
            mem_capacity[-1] += capL
            mem_capacity.append(capR)
    else:
        mem_capacity = _broadcast("mem_capacity", mem_capacity, n_nodes)

    qnodes: list[TopoQNode] = []
    for name, mem_capacity in zip(nodes, mem_capacity):
        qnodes.append(
            {
                "name": name,
                "memory": {
                    "t_cohere": t_cohere,
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

    path = RoutingPathSingle("S", "D", swap=swap, swap_cutoff=swap_cutoff)
    controller: TopoController = {
        "name": "ctrl",
        "apps": [ProactiveRoutingController(path)],
    }
    for node in nodes:
        cchannels.append({"node1": "ctrl", "node2": node, "parameters": {"delay": CTRL_DELAY}})

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
