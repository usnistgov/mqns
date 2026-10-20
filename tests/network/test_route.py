import itertools

import pytest

from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.network import QuantumNetwork
from mqns.network.route import DijkstraRouteAlgorithm, YenRouteAlgorithm
from mqns.network.topology.customtopo import CustomTopology


def test_dijkstra():
    net = QuantumNetwork(route=DijkstraRouteAlgorithm())
    net.add_node(n1 := QNode("n1"))
    net.add_node(n2 := QNode("n2"))
    net.add_node(n3 := QNode("n3"))
    net.add_node(n4 := QNode("n4"))
    for a, b in itertools.pairwise(net.nodes):
        ch = QuantumChannel(f"q_{a.name},{b.name}")
        a.add_qchannel(ch)
        b.add_qchannel(ch)
        net.add_qchannel(ch)
    net.build_route()

    with pytest.raises(match="no route"):
        net.query_route("n1", "n1")

    r11 = net.query_route("n1", "n1", error_on_empty=False)
    assert len(r11) == 0

    r12 = net.query_route("n1", "n2")
    assert len(r12) == 1
    assert r12[0].metric == 1
    assert r12[0].next_hop is n2
    assert r12[0].nodes == [n1, n2]
    assert r12[0].path == ["n1", "n2"]

    r14 = net.query_route("n1", "n4")
    assert len(r14) == 1
    assert r14[0].metric == 3
    assert r14[0].nodes == [n1, n2, n3, n4]
    assert r14[0].path == ["n1", "n2", "n3", "n4"]


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
    net = QuantumNetwork(topo, route=YenRouteAlgorithm(k_paths=3))
    net.build_route()

    routes = net.query_route("S", "D")

    print("\nComputed Yen paths from S to D:")
    for route in routes:
        print(f"  Cost: {route.metric}, Next hop: {route.next_hop.name}, Path: {route.path}")

    all_paths = [route.path for route in routes]

    # Assertions
    assert len(routes) >= 2
    assert ["S", "R1", "R2", "R3", "R4", "D"] in all_paths
    assert ["S", "R5", "R3", "R4", "D"] in all_paths
    for p in all_paths:
        assert len(p) == len(set(p)), f"Loop detected in path: {p}"
