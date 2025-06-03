
from qns.network.network import QuantumNetwork
from qns.network.topology import BasicTopology, GridTopology, LinearTopology, RandomTopology, TreeTopology, WaxmanTopology


def collect_qchannels(net: QuantumNetwork) -> set[tuple[int, int]]:
    result = set[tuple[int, int]]()
    for ch in net.qchannels:
        nA, nB = ch.node_list
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
        assert node.name == f"n{1+i}"

    assert len(net.qchannels) == 0

def test_linear_topo():
    """
    1---2---3---4
    """
    net = QuantumNetwork(topo=LinearTopology(4))

    assert len(net.nodes) == 4
    for i, node in enumerate(net.nodes):
        assert node.name == f"n{1+i}"

    assert len(net.qchannels) == 3
    for i, ch in enumerate(net.qchannels):
        assert set(ch.node_list) == set([net.nodes[i], net.nodes[i+1]])

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

    qchannels = collect_qchannels(net)
    assert qchannels == set([
        (0,1),(1,2),(3,4),(4,5),(6,7),(7,8), # horizontal
        (0,3),(1,4),(2,5),(3,6),(4,7),(5,8), # vertical
    ])

def test_grid_topo_rectangle():
    """
    1---2---3---4
    |   |   |   |
    5---6---7---8
    """
    net = QuantumNetwork(topo=GridTopology((2, 4)))

    assert len(net.nodes) == 8
    assert len(net.qchannels) == 10

    qchannels = collect_qchannels(net)
    assert qchannels == set([
        (0,1),(1,2),(2,3),(4,5),(5,6),(6,7), # horizontal
        (0,4),(1,5),(2,6),(3,7),             # vertical
    ])

def test_random_topo():
    net = QuantumNetwork(topo=RandomTopology(100, 400))

    assert len(net.nodes) == 100
    assert len(net.qchannels) == 400

    qchannels = collect_qchannels(net)
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

    qchannels = collect_qchannels(net)
    assert qchannels == set([
        (0,1),(0,2), # n1 to its children
        (1,3),(1,4), # n2 to its children
        (2,5),(2,6), # n3 to its children
    ])

def test_waxman_topo():
    net = QuantumNetwork(topo=WaxmanTopology(10, size=1000, alpha=0.5, beta=0.5))
    assert len(net.nodes) == 10
