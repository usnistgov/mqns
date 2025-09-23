from mqns.network.network import QuantumNetwork
from mqns.network.route import DijkstraRouteAlgorithm, YenRouteAlgorithm
from mqns.network.topology import CustomTopology, LinearTopology


def test_dijkstra():
    net = QuantumNetwork(topo=LinearTopology(4), route=DijkstraRouteAlgorithm())
    net.build_route()

    n1 = net.get_node("n1")
    n2 = net.get_node("n2")
    n3 = net.get_node("n3")
    n4 = net.get_node("n4")

    r11 = net.query_route(n1, n1)
    assert len(r11) == 0

    r12 = net.query_route(n1, n2)
    assert len(r12) == 1
    assert r12[0] == (1, n2, [n1, n2])

    r14 = net.query_route(n1, n4)
    assert len(r14) == 1
    assert r14[0] == (3, n2, [n1, n2, n3, n4])


def test_yen():
    """
    Test Yen's algorithm in the following topology:

      10    10    10    10    10
    S----R1----R2----R3----R4----D
    |                |
    +-------R5-------+
        15      5
    """
    topo = CustomTopology(
        {
            "qnodes": [
                {"name": name, "apps": [], "memory": {"capacity": 4}} for name in ["S", "R1", "R2", "R3", "R4", "R5", "D"]
            ],
            "qchannels": [
                {"node1": "S", "node2": "R1", "capacity1": 2, "capacity2": 2, "parameters": {"length": 10}},
                {"node1": "R1", "node2": "R2", "capacity1": 2, "capacity2": 2, "parameters": {"length": 10}},
                {"node1": "R2", "node2": "R3", "capacity1": 2, "capacity2": 1, "parameters": {"length": 10}},
                {"node1": "R3", "node2": "R4", "capacity1": 2, "capacity2": 2, "parameters": {"length": 10}},
                {"node1": "R4", "node2": "D", "capacity1": 2, "capacity2": 4, "parameters": {"length": 10}},
                {"node1": "S", "node2": "R5", "capacity1": 2, "capacity2": 2, "parameters": {"length": 15}},
                {"node1": "R5", "node2": "R3", "capacity1": 2, "capacity2": 1, "parameters": {"length": 5}},
            ],
            "cchannels": [],  # Classical links not required for this test
            "controller": {"name": "ctrl", "apps": []},
        }
    )
    net = QuantumNetwork(topo=topo, route=YenRouteAlgorithm(k_paths=3))
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
