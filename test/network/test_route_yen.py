from qns.network.network import QuantumNetwork
from qns.network.route import YenRouteAlgorithm
from qns.network.topology import CustomTopology


def generate_simple_topology() -> dict:
    return {
        "qnodes": [{"name": name, "apps": [], "memory": {}} for name in ["S", "R1", "R2", "R3", "R4", "R5", "D"]],
        "qchannels": [
            {"node1": "S",  "node2": "R1", "parameters": {"length": 10, "delay": 10 / 2e8}},
            {"node1": "R1", "node2": "R2", "parameters": {"length": 10, "delay": 10 / 2e8}},
            {"node1": "R2", "node2": "R3", "parameters": {"length": 10, "delay": 10 / 2e8}},
            {"node1": "R3", "node2": "R4", "parameters": {"length": 10, "delay": 10 / 2e8}},
            {"node1": "R4", "node2": "D",  "parameters": {"length": 10, "delay": 10 / 2e8}},
            {"node1": "S",  "node2": "R5", "parameters": {"length": 15, "delay": 15 / 2e8}},
            {"node1": "R5", "node2": "R3", "parameters": {"length": 5,  "delay": 5  / 2e8}}
        ],
        "cchannels": [],  # Classical links not required for this test
        "controller": {"name": "ctrl", "apps": []}
    }


def test_yen_routing():
    topo_dict = generate_simple_topology()
    topo = CustomTopology(topo_dict)
    net = QuantumNetwork(
        topo=topo,
        route=YenRouteAlgorithm(k_paths=3)
    )
    net.build_route()

    node_s = net.get_node("S")
    node_d = net.get_node("D")

    paths = net.query_route(node_s, node_d)

    print("\nComputed Yen paths from S to D:")
    for metric, next_hop, path in paths:
        print(f"  Cost: {metric}, Next hop: {next_hop.name}, Path: {[n.name for n in path]}")

    all_paths = [[n.name for n in p] for _, _, p in paths]

    # Assertions
    assert len(paths) >= 2
    assert ["S", "R1", "R2", "R3", "R4", "D"] in all_paths
    assert ["S", "R5", "R3", "R4", "D"] in all_paths
    for p in all_paths:
        assert len(p) == len(set(p)), f"Loop detected in path: {p}"

