from mqns.network.network import QuantumNetwork
from mqns.network.proactive import ProactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import log, set_seed

from examples_common.stats import gather_etg_decoh
from examples_common.topo_linear import build_topology

log.set_default_level("DEBUG")

SEED_BASE = 100

# parameters
sim_duration = 3

set_seed(SEED_BASE)

topo = build_topology(
    nodes=4,
    mem_capacity=4,  # number of qubits per node should be enough for qchannels
    t_cohere=0.01,  # sec
    channel_length=[32, 18, 10],
    channel_capacity=[(4, 3), (1, 2), (2, 4)],
    swap="swap_2_l2r",
)
net = QuantumNetwork(topo)

s = Simulator(0, sim_duration + 5e-06, accuracy=1000000, install_to=(log, net))
s.run()

#### get stats
_, _, decoh_ratio = gather_etg_decoh(net)
e2e_rate = net.get_node("S").get_app(ProactiveForwarder).cnt.n_consumed / sim_duration

print(f"E2E etg rate: {e2e_rate}")
print(f"Expired memories: {decoh_ratio}")
