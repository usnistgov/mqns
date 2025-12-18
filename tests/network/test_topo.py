from collections.abc import Sequence
from typing import cast

import pytest

from mqns.entity.cchannel import ClassicChannel
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.network import QuantumNetwork
from mqns.network.topology import (
    BasicTopology,
    ClassicTopology,
    CustomTopology,
    GridTopology,
    LinearTopology,
    RandomTopology,
    TreeTopology,
    WaxmanTopology,
)
from mqns.simulator import Simulator


def collect_channels(net: QuantumNetwork, channels: Sequence[ClassicChannel | QuantumChannel]) -> set[tuple[int, int]]:
    result = set[tuple[int, int]]()
    for ch in channels:
        nA, nB = cast(list[QNode], ch.node_list)
        iA, iB = net.nodes.index(nA), net.nodes.index(nB)
        assert iA != iB
        result.add((iA, iB) if iA < iB else (iB, iA))
    return result


def test_empty_net():
    net = QuantumNetwork()
    assert len(net.nodes) == 0
    assert len(net.qchannels) == 0


def test_basic_topo():
    net = QuantumNetwork(topo=BasicTopology(6))

    assert len(net.nodes) == 6
    for i, node in enumerate(net.nodes):
        assert node.name == f"n{1 + i}"

    assert len(net.qchannels) == 0


def test_linear_topo():
    """
    1---2---3---4
    """
    net = QuantumNetwork(topo=LinearTopology(4))

    assert len(net.nodes) == 4
    for i, node in enumerate(net.nodes):
        assert node.name == f"n{1 + i}"

    assert len(net.qchannels) == 3
    for i, ch in enumerate(net.qchannels):
        assert set(ch.node_list) == set([net.nodes[i], net.nodes[i + 1]])


def test_grid_topo_square():
    """
    1---2---3
    |   |   |
    4---5---6
    |   |   |
    7---8---9
    """
    net = QuantumNetwork(topo=GridTopology(9))

    assert len(net.nodes) == 9
    assert len(net.qchannels) == 12

    qchannels = collect_channels(net, net.qchannels)
    assert qchannels == set(
        [
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),  # horizontal
            (0, 3),
            (1, 4),
            (2, 5),
            (3, 6),
            (4, 7),
            (5, 8),  # vertical
        ]
    )


def test_grid_topo_rectangle():
    """
    1---2---3---4
    |   |   |   |
    5---6---7---8
    """
    net = QuantumNetwork(topo=GridTopology((2, 4)))

    assert len(net.nodes) == 8
    assert len(net.qchannels) == 10

    qchannels = collect_channels(net, net.qchannels)
    assert qchannels == set(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (4, 5),
            (5, 6),
            (6, 7),  # horizontal
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # vertical
        ]
    )


def test_random_topo():
    net = QuantumNetwork(topo=RandomTopology(100, 400))

    assert len(net.nodes) == 100
    assert len(net.qchannels) == 400

    qchannels = collect_channels(net, net.qchannels)
    assert len(qchannels) == 400


def test_tree_topo():
    """
             +--4
       +--2--+
       |     +--5
    1--+
       |     +--6
       +--3--+
             +--7
    """
    net = QuantumNetwork(topo=TreeTopology(7, 2))

    assert len(net.nodes) == 7
    assert len(net.qchannels) == 6

    qchannels = collect_channels(net, net.qchannels)
    assert qchannels == set(
        [
            (0, 1),
            (0, 2),  # n1 to its children
            (1, 3),
            (1, 4),  # n2 to its children
            (2, 5),
            (2, 6),  # n3 to its children
        ]
    )


def test_waxman_topo():
    net = QuantumNetwork(topo=WaxmanTopology(10, size=1000, alpha=0.5, beta=0.5))
    assert len(net.nodes) == 10


def test_classical_empty():
    net = QuantumNetwork(topo=GridTopology(9), classic_topo=ClassicTopology.Empty)
    assert len(net.cchannels) == 0


def test_classical_all():
    n = 9
    net = QuantumNetwork(topo=GridTopology(n), classic_topo=ClassicTopology.All)

    cchannels = collect_channels(net, net.cchannels)
    assert len(cchannels) == n * (n - 1) // 2


def test_classical_follow():
    net = QuantumNetwork(topo=GridTopology(9), classic_topo=ClassicTopology.Follow)

    qchannels = collect_channels(net, net.qchannels)
    cchannels = collect_channels(net, net.cchannels)
    assert cchannels == qchannels


def test_custom_topo_empty():
    topo = CustomTopology(
        topo={
            "qnodes": [],
            "qchannels": [],
            "cchannels": [],
        }
    )
    net = QuantumNetwork(topo=topo)
    net.install(Simulator())

    assert len(net.nodes) == 0
    assert len(net.qchannels) == 0
    assert len(net.cchannels) == 0
    assert net.controller is None
    assert len(net.all_nodes) == 0


def test_custom_topo_basic():
    topo = CustomTopology(
        topo={
            "qnodes": [
                {"name": "A", "memory": {"capacity": 2}, "apps": []},
                {"name": "B", "memory": {"capacity": 4}, "apps": []},
                {"name": "C", "memory": {"capacity": 3}, "apps": []},
            ],
            "qchannels": [
                {"node1": "A", "node2": "B", "capacity1": 2, "capacity2": 1, "parameters": {}},
                {"node1": "B", "node2": "C", "capacity": 3, "parameters": {}},
            ],
            "cchannels": [
                {"node1": "A", "node2": "C", "parameters": {}},
            ],
            "controller": {
                "name": "ctrl",
                "apps": [],
            },
        }
    )
    net = QuantumNetwork(topo=topo)
    net.install(Simulator())

    assert len(net.nodes) == 3
    assert len(net.qchannels) == 2
    assert len(net.cchannels) == 1
    assert net.controller is not None
    assert len(net.all_nodes) == 4

    mA = net.get_node("A").memory
    mB = net.get_node("B").memory
    mC = net.get_node("C").memory

    qAB = net.get_qchannel("q_A,B")
    qBC = net.get_qchannel("q_B,C")

    assert mA.capacity == 2
    assert mA.read(0, must=True)[0].qchannel == qAB
    assert mA.read(1, must=True)[0].qchannel == qAB
    assert mB.capacity == 4
    assert mB.read(0, must=True)[0].qchannel == qAB
    assert mB.read(1, must=True)[0].qchannel == qBC
    assert mB.read(2, must=True)[0].qchannel == qBC
    assert mB.read(3, must=True)[0].qchannel == qBC
    assert mC.capacity == 3
    assert mC.read(0, must=True)[0].qchannel == qBC
    assert mC.read(1, must=True)[0].qchannel == qBC
    assert mC.read(2, must=True)[0].qchannel == qBC


def test_custom_topo_low_memory():
    topo = CustomTopology(
        topo={
            "qnodes": [
                {"name": "A", "memory": {"capacity": 1}, "apps": []},
                {"name": "B", "memory": {"capacity": 2}, "apps": []},
            ],
            "qchannels": [
                {"node1": "A", "node2": "B", "capacity": 2, "parameters": {}},
            ],
            "cchannels": [],
        }
    )
    with pytest.raises(OverflowError, match="insufficient qubits"):
        _ = QuantumNetwork(topo=topo)
