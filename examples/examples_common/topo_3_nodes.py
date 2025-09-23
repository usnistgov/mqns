from mqns.network.proactive import LinkLayer, ProactiveForwarder, ProactiveRoutingController, RoutingPathSingle
from mqns.network.topology.customtopo import CustomTopology, Topo
from mqns.network.topology.topo import Topology


def build_topology(
    *,
    t_coherence: float,
    channel_qubits: int,
    ch_1: float = 32,
    ch_2: float = 18,
    entg_attempt_rate: float = 50e6,  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)
    init_fidelity: float = 0.99,
    fiber_alpha: float = 0.2,
    eta_d: float = 0.95,
    eta_s: float = 0.95,
    frequency: float = 1e6,  # memory frequency
    p_swap: float = 0.5,
    swapping_order: str = "swap_1",
) -> Topology:
    """
    Build a linear topology consists of three nodes S-R-D.
    This topology is illustrated in `examples/images/3_nodes_thruput.png`.

    Args:
        # QuantumMemory
        t_coherence: memory coherence time in seconds.
        # QuantumChannel
        channel_qubits: qchannel capacity.
        ch_1: S-R qchannel length.
        ch_2: R-D qchannel length.
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
        swapping_order: named swapping order.
    """
    topo: Topo = {
        "qnodes": [
            {
                "name": "S",
                "memory": {"decoherence_rate": 1 / t_coherence, "capacity": channel_qubits},
                "apps": [
                    LinkLayer(
                        attempt_rate=entg_attempt_rate,
                        init_fidelity=init_fidelity,
                        alpha_db_per_km=fiber_alpha,
                        eta_d=eta_d,
                        eta_s=eta_s,
                        frequency=frequency,
                    ),
                    ProactiveForwarder(),
                ],
            },
            {
                "name": "R",
                "memory": {"decoherence_rate": 1 / t_coherence, "capacity": channel_qubits * 2},
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
            },
            {
                "name": "D",
                "memory": {"decoherence_rate": 1 / t_coherence, "capacity": channel_qubits},
                "apps": [
                    LinkLayer(
                        attempt_rate=entg_attempt_rate,
                        init_fidelity=init_fidelity,
                        alpha_db_per_km=fiber_alpha,
                        eta_d=eta_d,
                        eta_s=eta_s,
                        frequency=frequency,
                    ),
                    ProactiveForwarder(),
                ],
            },
        ],
        "qchannels": [
            {
                "node1": "S",
                "node2": "R",
                "capacity": channel_qubits,
                "parameters": {"length": ch_1},
            },
            {
                "node1": "R",
                "node2": "D",
                "capacity": channel_qubits,
                "parameters": {"length": ch_2},
            },
        ],
        "cchannels": [
            {"node1": "S", "node2": "R", "parameters": {"length": ch_1}},
            {"node1": "R", "node2": "D", "parameters": {"length": ch_2}},
            {"node1": "ctrl", "node2": "S", "parameters": {"length": 1.0}},
            {"node1": "ctrl", "node2": "R", "parameters": {"length": 1.0}},
            {"node1": "ctrl", "node2": "D", "parameters": {"length": 1.0}},
        ],
        "controller": {
            "name": "ctrl",
            "apps": [ProactiveRoutingController(RoutingPathSingle("S", "D", swap=swapping_order))],
        },
    }
    return CustomTopology(topo)
