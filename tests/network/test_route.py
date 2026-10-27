import itertools

import pytest

from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.network import QuantumNetwork
from mqns.network.route import DijkstraRouteAlgorithm, MetricFunc, YenRouteAlgorithm


def test_dijkstra():
    net = QuantumNetwork(route=DijkstraRouteAlgorithm())
    net.add_node(n1 := QNode("n1"))
    net.add_node(n2 := QNode("n2"))
    net.add_node(n3 := QNode("n3"))
    net.add_node(n4 := QNode("n4"))
    for a, b in itertools.pairwise(net.nodes):
        net.add_qchannel(QuantumChannel(f"q_{a.name},{b.name}"), a, b)
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


@pytest.mark.parametrize(
    ("metric_func", "cost1", "cost5"),
    [
        pytest.param(None, 5, 4, id="metric_hops"),
        pytest.param(lambda ch: ch.length, 50, 40, id="metric_length"),
    ],
)
def test_yen(metric_func: MetricFunc, cost1: float, cost5: float):
    """
    Test Yen's algorithm in the following topology:

      10    10    10    10    10
    S----R1----R2----R3----R4----D
    |                |
    +-------R5-------+
        15      5
    """
    net = QuantumNetwork(route=YenRouteAlgorithm(k_paths=3, metric_func=metric_func))
    net.add_node(nodeS := QNode("S"))
    net.add_node(nodeD := QNode("D"))
    net.add_node(nodeR1 := QNode("R1"))
    net.add_node(nodeR2 := QNode("R2"))
    net.add_node(nodeR3 := QNode("R3"))
    net.add_node(nodeR4 := QNode("R4"))
    net.add_node(nodeR5 := QNode("R5"))
    net.add_qchannel(QuantumChannel("S-R1", length=10), nodeS, nodeR1)
    net.add_qchannel(QuantumChannel("R1-R2", length=10), nodeR1, nodeR2)
    net.add_qchannel(QuantumChannel("R2-R3", length=10), nodeR2, nodeR3)
    net.add_qchannel(QuantumChannel("R3-R4", length=10), nodeR3, nodeR4)
    net.add_qchannel(QuantumChannel("R4-D", length=10), nodeR4, nodeD)
    net.add_qchannel(QuantumChannel("S-R5", length=15), nodeS, nodeR5)
    net.add_qchannel(QuantumChannel("R5-R3", length=5), nodeR5, nodeR3)
    net.build_route()

    routes = net.query_route("S", "D")

    assert len(routes) == 2
    route5, route1 = routes
    assert cost1 > cost5  # route5 ranks before route1

    assert route1.metric == pytest.approx(cost1, abs=1e-6)
    assert route1.path == ["S", "R1", "R2", "R3", "R4", "D"]

    assert route5.metric == pytest.approx(cost5, abs=1e-6)
    assert route5.path == ["S", "R5", "R3", "R4", "D"]
