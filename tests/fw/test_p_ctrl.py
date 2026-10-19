"""
Test suite for proactive forwarding focused on control plane.
"""

import copy
from collections.abc import Mapping
from typing import NamedTuple

import pytest

from mqns.network.builder import GridTopoGen, NetworkBuilder
from mqns.network.fw import RoutingController, RoutingPath
from mqns.network.fw.message import PathInstructions
from mqns.network.network import QuantumNetwork, Request, RequestState
from mqns.utils import unwrap

from .fw_common import make_network


class _PathInsertRecord(NamedTuple):
    t: float
    insts: list[PathInstructions]
    route: list[str]


def collect_path_insert(monkeypatch: pytest.MonkeyPatch, net: QuantumNetwork) -> Mapping[int, _PathInsertRecord]:
    ctrl = unwrap(net.controller).get_app(RoutingController)
    old_path_insert = ctrl.path_insert
    records: dict[int, _PathInsertRecord] = {}

    def new_path_insert(req_id: int, insts: list[PathInstructions], **kwargs):
        records[req_id] = _PathInsertRecord(net.simulator.tc.sec, insts, ["".join(inst["route"]) for inst in insts])
        return old_path_insert(req_id, insts, **kwargs)

    monkeypatch.setattr(ctrl, "path_insert", new_path_insert)
    return records


def test_rect2_ru():
    """Test resource utilization tracking in 2x2 rectangular topology."""
    net, simulator = make_network(
        NetworkBuilder()
        .topo_from(GridTopoGen((2, 2)), nodes_naming="A", t_cohere=0.005, ch_capacity=3)
        .proactive_centralized(drq_cap=0),
    )
    # FIB erase delay is 4*t_cohere = 0.020 seconds.

    # t=0.000, available qubits per link: AB=3, AC=3, BD=3, CD=3.
    net.add_request(req0 := Request(RoutingPath.static("ABD", "ACD", bufferspace_mv=1), active_period=(1.001, 2.001)))
    # t=1.001, available qubits per link: AB=2, AC=2, BD=2, CD=2.
    net.add_request(req1 := Request(RoutingPath.static("ABD", bufferspace_mv=1), active_period=(1.002, 2.002)))
    # t=1.002, available qubits per link: AB=0, AC=2, BD=0, CD=2.
    net.add_request(req2 := Request(RoutingPath.static("ABD", bufferspace_mv=2), active_period=(1.003, 2.003)))
    # req2 is rejected due to insufficient resources on AB and BD links.
    net.add_request(req3 := Request(RoutingPath.static("ACD", bufferspace_mv=2), active_period=(1.004, 2.004)))
    # t=1.004, available qubits per link: AB=0, AC=0, BD=0, CD=0.

    # req0 releases its qubits at 2.021; req3 releases its qubits at 2.024.
    # t=2.023, req0,req1,req2 released, available qubits per link: AB=3, AC=1, BD=3, CD=1.
    net.add_request(req4 := Request(RoutingPath.static("BAC", bufferspace_mv=3), active_period=(2.023, 3.023)))
    # req4 is rejected due to insufficient resources on AC link.
    # t=2.024, req3 released, available qubits per link: AB=3, AC=3, BD=3, CD=3.
    net.add_request(req5 := Request(RoutingPath.static("AC", bufferspace_mv=3), active_period=(2.025, 3.025)))

    simulator.run()

    assert req0.state is RequestState.EXPIRED
    assert req1.state is RequestState.EXPIRED
    assert req2.state is RequestState.REJECTED
    assert req3.state is RequestState.EXPIRED
    assert req4.state is RequestState.REJECTED
    assert req5.state is RequestState.EXPIRED


@pytest.mark.parametrize(
    ("mv2", "state3"),
    [
        # t=1.620, req1 released, available qubits per link: AB=1, AC=2, BD=1, CD=2.
        # req3 is accepted at this time.
        (2, RequestState.EXPIRED),
        # t=1.620, req1 released, available qubits per link: AB=0, AC=2, BD=0, CD=2.
        # There's still no resources to accept req3.
        # t=2.004, req3 is rejected when it reaches expiration.
        (3, RequestState.REJECTED),
    ],
)
def test_rect2_drq(monkeypatch: pytest.MonkeyPatch, mv2: int, state3: RequestState):
    """Test deferred request queue in 2x2 rectangular topology."""
    net, simulator = make_network(
        NetworkBuilder()
        .topo_from(GridTopoGen((2, 2)), nodes_naming="A", t_cohere=0.005, ch_capacity=3)
        .proactive_centralized(drq_cap=2),
    )
    # FIB erase delay is 4*t_cohere = 0.020 seconds.
    inserted_paths = collect_path_insert(monkeypatch, net)

    # t=0.000, available qubits per link: AB=3, AC=3, BD=3, CD=3.
    net.add_request(req0 := Request(RoutingPath.static("ABD", bufferspace_mv=3), active_period=(1.001, 1.500)))
    net.add_request(req1 := Request(RoutingPath.static("ACD", bufferspace_mv=2), active_period=(1.002, 1.600)))
    # t=1.002, available qubits per link: AB=0, AC=1, BD=0, CD=1.
    net.add_request(req2 := Request(RoutingPath.static("ABD", bufferspace_mv=mv2), active_period=(1.003, 2.003)))
    # req2 is deferred due to insufficient resources on AB and BD links.
    net.add_request(req3 := Request(RoutingPath.static("ABD", "ACD", bufferspace_mv=1), active_period=(1.004, 2.004)))
    # req3 is deferred due to insufficient resources on AB and BD links.
    net.add_request(req4 := Request(RoutingPath.static("ABD", bufferspace_mv=1), active_period=(1.005, 2.005)))
    # req4 is rejected because the queue is full.
    net.add_request(req5 := Request(RoutingPath.static("ACD", bufferspace_mv=1), active_period=(1.006, 2.006)))
    # t=1.006, available qubits per link: AB=0, AC=0, BD=0, CD=0.

    # t=1.520, req0 released, available qubits per link: AB=3, AC=0, BD=3, CD=0.
    # req2 is accepted at this time.
    # See parametrize comments for fate of req3.

    simulator.run()

    assert req0.state is RequestState.EXPIRED
    assert inserted_paths[req0.req_id].t == pytest.approx(1.001, abs=1e-6)
    assert req1.state is RequestState.EXPIRED
    assert inserted_paths[req1.req_id].t == pytest.approx(1.002, abs=1e-6)
    assert req2.state is RequestState.EXPIRED
    assert inserted_paths[req2.req_id].t == pytest.approx(1.520, abs=1e-6)
    assert req3.state is state3
    if state3 is RequestState.EXPIRED:
        assert inserted_paths[req3.req_id].t == pytest.approx(1.620, abs=1e-6)
    else:
        assert req3.req_id not in inserted_paths
    assert req4.state is RequestState.REJECTED
    assert req4.req_id not in inserted_paths
    assert req5.state is RequestState.EXPIRED
    assert inserted_paths[req5.req_id].t == pytest.approx(1.006, abs=1e-6)


def test_grid12_multipath(monkeypatch: pytest.MonkeyPatch):
    """
    Test multi-path reroute in 12-node (3 rows, 4 columns) grid topology.

        A---B---C-x-D
        |   |   |   |
        E---F---G---H
        |   |   x   |
        I---J---K---L
    """
    net, simulator = make_network(
        NetworkBuilder().topo_from(GridTopoGen((3, 4)), nodes_naming="A", ch_capacity=1).proactive_centralized(),
    )
    inserted_paths = collect_path_insert(monkeypatch, net)

    # Block off CD and GK links, marked "x" in the diagram.
    net.add_request(Request(RoutingPath.static("CD"), active_period=(1.000, 4.000)))
    net.add_request(Request(RoutingPath.static("GK"), active_period=(1.000, 4.000)))

    rp = RoutingPath.static("FG", "FEABCG", "FBCDHG", "FEIJKG", "FJKLHG", multipath="any", bufferspace_mv=1)
    net.add_request(req0 := Request(copy.deepcopy(rp), active_period=(1.001, 2.001)))
    # req0 uses FG path.
    net.add_request(req1 := Request(copy.deepcopy(rp), active_period=(1.002, 2.002)))
    # req1 uses FEABCG path.
    net.add_request(req2 := Request(copy.deepcopy(rp), active_period=(1.003, 2.003)))
    # req2 uses FJKLHG path.
    net.add_request(req3 := Request(copy.deepcopy(rp), active_period=(1.004, 2.004)))
    net.add_request(req4 := Request(copy.deepcopy(rp), active_period=(1.005, 2.005)))
    # req3 and req4 are rejected.

    simulator.run()

    assert req0.state is RequestState.EXPIRED
    assert inserted_paths[req0.req_id].route == ["FG"]
    assert req1.state is RequestState.EXPIRED
    assert inserted_paths[req1.req_id].route == ["FEABCG"]
    assert req2.state is RequestState.EXPIRED
    assert inserted_paths[req2.req_id].route == ["FJKLHG"]
    assert req3.state is RequestState.REJECTED
    assert req3.req_id not in inserted_paths
    assert req4.state is RequestState.REJECTED
    assert req4.req_id not in inserted_paths
