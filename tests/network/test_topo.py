from collections.abc import Iterable
from typing import cast

import pytest

from mqns.entity.cchannel import ClassicChannel
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.network import QuantumNetwork
from mqns.network.topology import (
    CustomTopology,
)
from mqns.simulator import Simulator


def collect_channels(net: QuantumNetwork, channels: Iterable[ClassicChannel | QuantumChannel]) -> set[tuple[int, int]]:
    nl = list(net.nodes)
    result = set[tuple[int, int]]()
    for ch in channels:
        nA, nB = cast(list[QNode], ch.node_list)
        iA, iB = nl.index(nA), nl.index(nB)
        assert iA != iB
        result.add((iA, iB) if iA < iB else (iB, iA))
    return result


def test_empty_net():
    net = QuantumNetwork()
    assert len(net.nodes) == 0
    assert len(net.qchannels) == 0


def test_custom_topo_empty():
    topo = CustomTopology(
        topo={
            "qnodes": [],
            "qchannels": [],
            "cchannels": [],
        }
    )
    net = QuantumNetwork(topo)
    net.install(Simulator(0, 10))

    assert len(net.nodes) == 0
    assert len(net.qchannels) == 0
    assert len(net.cchannels) == 0
    assert net.controller is None
    assert len([*net.all_nodes]) == 0


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
    net = QuantumNetwork(topo)
    net.install(Simulator(0, 10))

    assert [node.name for node in net.nodes] == ["A", "B", "C"]
    assert len(net.qchannels) == 2
    assert len(net.cchannels) == 1
    assert net.controller is not None
    assert [node.name for node in net.all_nodes] == ["A", "B", "C", "ctrl"]

    mA = net.get_node("A").memory
    mB = net.get_node("B").memory
    mC = net.get_node("C").memory

    qAB = net.get_qchannel("A", "B")
    qBC = net.get_qchannel("B", "C")

    assert mA.capacity == 2
    assert mA.read(0)[0].qchannel is qAB
    assert mA.read(1)[0].qchannel is qAB
    assert mB.capacity == 4
    assert mB.read(0)[0].qchannel is qAB
    assert mB.read(1)[0].qchannel is qBC
    assert mB.read(2)[0].qchannel is qBC
    assert mB.read(3)[0].qchannel is qBC
    assert mC.capacity == 3
    assert mC.read(0)[0].qchannel is qBC
    assert mC.read(1)[0].qchannel is qBC
    assert mC.read(2)[0].qchannel is qBC


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
        _ = QuantumNetwork(topo)
