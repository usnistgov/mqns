import itertools
from collections.abc import Sequence

import pytest

from mqns.network.builder import ChannelDef, ChannelParam, NetworkBuilder, NodeDef
from mqns.network.network import TimingMode, TimingModeSync
from mqns.network.protocol.link_layer import LinkLayer
from mqns.simulator import Simulator


def test_errors():
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

    with pytest.raises(ValueError, match="incorrect number of channels"):
        NetworkBuilder().topo_linear(nodes=4, channels=[10, 20])

    with pytest.raises(NotImplementedError):
        NetworkBuilder().topo_linear(nodes=2).proactive_distributed()

    with pytest.raises(NotImplementedError):
        NetworkBuilder().topo_linear(nodes=2).reactive_distributed()


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


def test_topo_linear_numbered():
    b = NetworkBuilder()
    b.topo_linear(
        nodes=4,
        channels=[10, 20, 30],
    )
    b.proactive_centralized()
    net = b.make_network()

    node_names = ["S", "R1", "R2", "D"]
    assert [node.name for node in net.nodes] == node_names
    assert [tuple(node.name for node in ch.node_list) for ch in net.qchannels] == list(itertools.pairwise(node_names))
    assert [ch.length for ch in net.qchannels] == [10, 20, 30]


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
