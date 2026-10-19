import itertools
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from mqns.network.builder import ChannelDef, ChannelParam, GridTopoGen, NetworkBuilder, NodeDef, TreeTopoGen
from mqns.network.network import TimingMode, TimingModeSync
from mqns.network.protocol.link_layer import LinkLayer
from mqns.simulator import Simulator


def test_errors_builder():
    with pytest.raises(TypeError, match="must define topology first"):
        NetworkBuilder().proactive_centralized()

    with pytest.raises(TypeError, match="applications already installed"):
        NetworkBuilder().topo_linear(nodes=2).proactive_centralized().proactive_centralized()

    with pytest.raises(TypeError, match="must install controller application"):
        NetworkBuilder().topo_linear(nodes=3).request("S-D")

    with pytest.raises(TypeError, match="must install applications"):
        NetworkBuilder().topo_linear(nodes=2).make_network()

    with pytest.raises(ValueError, match="at least two nodes"):
        NetworkBuilder().topo_linear(nodes=1)

    with pytest.raises(ValueError, match="at least two nodes"):
        NetworkBuilder().topo_linear(nodes=["A"])

    with pytest.raises(ValueError, match="too many nodes for nodes_naming='A'"):
        NetworkBuilder().topo_linear(nodes=27, nodes_naming="A")

    with pytest.raises(ValueError, match="incorrect number of channels"):
        NetworkBuilder().topo_linear(nodes=4, channels=[10, 20])

    with pytest.raises(NotImplementedError):
        NetworkBuilder().topo_linear(nodes=2).proactive_distributed()

    with pytest.raises(NotImplementedError):
        NetworkBuilder().topo_linear(nodes=2).reactive_distributed()


def test_errors_topogen():
    with pytest.raises(ValueError, match="every dimension of a grid topology must be positive"):
        GridTopoGen((0, 1))

    with pytest.raises(ValueError, match="arity must be nonnegative"):
        TreeTopoGen(size=4, arity=-1)

    with pytest.raises(ValueError, match="a tree with arity 0 can only have size=1 or levels=1"):
        TreeTopoGen(size=4, arity=0)

    with pytest.raises(ValueError, match="a tree with arity 0 can only have size=1 or levels=1"):
        TreeTopoGen(levels=2, arity=0)

    with pytest.raises(ValueError, match="levels must be positive"):
        TreeTopoGen(levels=0, arity=2)

    with pytest.raises(ValueError, match="size must be positive"):
        TreeTopoGen(size=0, arity=2)


def test_topo_manual():
    b = NetworkBuilder()
    b.topo(
        channels=[
            ("A-B", 10),
            ("B-C", 20, 2),
            ("C-D", 30, (2, 3)),
            ChannelDef("D-A", fiber_alpha=0.4),
        ],
        nodes=[
            NodeDef("A", mem_capacity=9, tau_0=0.1),
        ],
        ch_length=80,
        fiber_alpha=0.1,
    )
    b.proactive_centralized()
    net = b.make_network()

    nodes = sorted(net.nodes, key=lambda node: node.name)
    assert [node.name for node in nodes] == ["A", "B", "C", "D"]
    assert [node.memory.capacity for node in nodes] == [9, 1 + 2, 2 + 2, 3 + 1]
    assert [node.get_app(LinkLayer).tau_0 for node in nodes] == pytest.approx([0.1, 0.0, 0.0, 0.0], abs=1e-6)
    channels = sorted(net.qchannels, key=lambda ch: ch.length)
    assert [ch.length for ch in channels] == [10, 20, 30, 80]
    assert [ch.alpha for ch in channels] == pytest.approx([0.1, 0.1, 0.1, 0.4], abs=1e-6)


def test_topo_grid():
    b = NetworkBuilder()
    b.topo_from(GridTopoGen((2, 3)), nodes_naming="A")
    b.proactive_centralized()
    net = b.make_network()

    # A---B---C
    # |   |   |
    # D---E---F
    channels = {tuple(node.name for node in ch.node_list) for ch in net.qchannels}
    assert channels == {("A", "B"), ("B", "C"), ("D", "E"), ("E", "F"), ("A", "D"), ("B", "E"), ("C", "F")}


@pytest.mark.parametrize(
    ("gen", "size", "links"),
    [
        pytest.param(lambda: TreeTopoGen(size=1, arity=0), 1, "", id="single"),
        pytest.param(lambda: TreeTopoGen(size=4, arity=1), 4, "AB,BC,CD", id="linear-size"),
        pytest.param(lambda: TreeTopoGen(levels=4, arity=1), 4, "AB,BC,CD", id="linear-levels"),
        pytest.param(lambda: TreeTopoGen(levels=3, arity=3), 13, "AB,AC,AD,BE,BF,BG,CH,CI,CJ,DK,DL,DM", id="ternary-13"),
        pytest.param(lambda: TreeTopoGen(size=8, arity=3), 8, "AB,AC,AD,BE,BF,BG,CH", id="ternary-8"),
    ],
)
def test_topo_tree(gen: Callable[[], TreeTopoGen], size: int, links: str):
    b = NetworkBuilder()
    b.topo_from(gen(), nodes_naming="A", mem_capacity=4)
    b.proactive_centralized()
    net = b.make_network()

    assert len(net.nodes) == size
    if size == 1:
        assert len(net.qchannels) == 0
    else:
        channels = {tuple(node.name for node in ch.node_list) for ch in net.qchannels}
        assert channels == set(tuple(np) for np in links.split(","))


@pytest.mark.parametrize(
    ("nodes_naming", "node_names"),
    [
        ("R1", ["S", "R1", "D"]),
        ("R", ["S", "R", "D"]),
        ("n1", ["n1", "n2", "n3"]),
        ("A", ["A", "B", "C"]),
        ("R1", ["S", "R1", "R2", "R3", "D"]),
        ("R", ["S", "R1", "R2", "R3", "D"]),
        ("n1", ["n1", "n2", "n3", "n4", "n5"]),
        ("A", ["A", "B", "C", "D", "E"]),
        ("A", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    ],
)
def test_topo_linear_numbered(nodes_naming: Any, node_names: list[str]):
    n_nodes = len(node_names)
    ch_lengths = list(range(10, 10 * n_nodes, 10))

    b = NetworkBuilder()
    b.topo_linear(
        nodes=n_nodes,
        nodes_naming=nodes_naming,
        channels=ch_lengths,
    )
    b.proactive_centralized()
    net = b.make_network()

    assert [node.name for node in net.nodes] == node_names
    assert [tuple(node.name for node in ch.node_list) for ch in net.qchannels] == list(itertools.pairwise(node_names))
    assert [ch.length for ch in net.qchannels] == ch_lengths


def test_topo_linear_named():
    b = NetworkBuilder()
    b.topo_linear(
        nodes=[NodeDef("A", tau_0=0.1), "B", "C", "D", "E"],
        channels=[
            10,  # A-B
            ChannelParam(fiber_alpha=0.4),  # B-C
            (),  # C-D
            (20, 3),  # D-E
        ],
        ch_length=80,
        fiber_alpha=0.1,
    )
    b.proactive_centralized()
    net = b.make_network()

    assert [node.name for node in net.nodes] == ["A", "B", "C", "D", "E"]
    assert [ch.length for ch in net.qchannels] == [10, 80, 80, 20]
    assert [ch.alpha for ch in net.qchannels] == pytest.approx([0.1, 0.4, 0.1, 0.1], abs=1e-6)


@pytest.mark.parametrize(
    "timing_input",
    [
        pytest.param(TimingModeSync(t_ext=0.007, t_rtg=0.001, t_int=0.002), id="instance"),
        pytest.param((0.007, 0.001, 0.002), id="tuple"),
    ],
)
def test_timing_explicit(timing_input: TimingMode | Sequence[float]):
    b = NetworkBuilder().topo_linear(nodes=3)
    b.proactive_centralized(timing=timing_input)
    net = b.make_network()
    Simulator(0, 10, accuracy=1000000, install_to=(net,))

    timing = net.timing
    assert isinstance(timing, TimingModeSync)
    assert timing.t_ext.slot == 7000
    assert timing.t_rtg.slot == 1000
    assert timing.t_int.slot == 2000


def test_timing_derived():
    b = NetworkBuilder().topo_linear(nodes=3, t_cohere=0.010)
    b.proactive_centralized(timing=())
    net = b.make_network()
    Simulator(0, 10, accuracy=1000000, install_to=(net,))

    timing = net.timing
    assert isinstance(timing, TimingModeSync)
    assert timing.t_ext.slot == 4990
    assert timing.t_rtg.slot == 20
    assert timing.t_int.slot == 4990
