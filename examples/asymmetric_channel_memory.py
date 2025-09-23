import logging

from mqns.network.network import QuantumNetwork
from mqns.network.proactive import ProactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from examples_common.stats import gather_etg_decoh
from examples_common.topo_asymmetric_channel import build_topology

log.logger.setLevel(logging.DEBUG)

SEED_BASE = 100

# parameters
sim_duration = 3

set_seed(SEED_BASE)
s = Simulator(0, sim_duration + 5e-06, accuracy=1000000)
log.install(s)

topo = build_topology(
    nodes=["S", "R1", "R2", "D"],
    mem_capacities=[4, 4, 4, 4],  # number of qubits per node should be enough for qchannels
    ch_lengths=[32, 18, 10],
    ch_capacities=[(4, 3), (1, 2), (2, 4)],
    t_coherence=0.01,  # sec
    swapping_order="swap_2_l2r",
)
net = QuantumNetwork(topo=topo)
net.install(s)

s.run()

#### get stats
_, _, decoh_ratio = gather_etg_decoh(net)
e2e_rate = net.get_node("S").get_app(ProactiveForwarder).cnt.n_consumed / sim_duration

print(f"E2E etg rate: {e2e_rate}")
print(f"Expired memories: {decoh_ratio}")
