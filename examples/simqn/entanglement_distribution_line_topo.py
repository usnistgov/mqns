import logging

import numpy as np

from mqns.network import QuantumNetwork
from mqns.network.protocol.entanglement_distribution import EntanglementDistributionApp
from mqns.network.route.dijkstra import DijkstraRouteAlgorithm
from mqns.network.topology import LineTopology
from mqns.network.topology.topo import ClassicTopology
from mqns.simulator.simulator import Simulator
from mqns.utils import log

log.logger.setLevel(logging.DEBUG)

light_speed = 299791458


def drop_rate(length):
    # drop 0.2 db/KM
    return 1 - np.exp(-length / 50000)


# constrains
init_fidelity = 0.99
nodes_number = 10
link_length = 10
memory_capacity = 50
send_rate = 10

result = []

s = Simulator(0, 10, accuracy=10000000)
log.install(s)
topo = LineTopology(
    nodes_number=nodes_number,
    qchannel_args={"delay": link_length / light_speed, "drop_rate": drop_rate(link_length)},
    cchannel_args={"delay": link_length / light_speed},
    memory_args=[{"capacity": memory_capacity, "decoherence_rate": 0.1}],
    nodes_apps=[EntanglementDistributionApp(init_fidelity=init_fidelity)],
)

net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.All, route=DijkstraRouteAlgorithm())
net.build_route()

src = net.get_node("n1")
dst = net.get_node(f"n{nodes_number}")
net.add_request(src=src, dest=dst, attr={"send_rate": send_rate})
net.install(s)
s.run()
result.append(dst.apps[-1].success[0].fidelity)
log.monitor(f"{nodes_number} {result}")
