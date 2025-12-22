import configparser
import csv
import os

from src.simulate_mqsn_eval import simulate
from src.utils.utils import generate_sim_id

config_path = os.path.join("data", "configs", "common_parameters.ini")
common_parameters = configparser.ConfigParser()
common_parameters.read(config_path)

network_sizes: list[tuple[int, int]] = [(16, 20), (32, 40), (64, 80), (128, 160), (256, 320), (512, 640)]


# parameters
sim_duration = 3

fiber_alpha = 0.2
eta_d = 0.95
eta_s = 0.95
frequency = 1e6  # memory frequency
entg_attempt_rate = 50e6  # From fiber max frequency (50 MHz) AND detectors count rate (60 MHz)

init_fidelity = 0.99
t_coherence = 5e-3  # 10e-3

p_swap = 0.5
swapping_policy = "ASAP"

nqubits = 2000  # large enough to support qchannel capacity in random topology
##########
qchannel_capacity = 100  # full simulation tries 10, 50, 100 qubits
##########
edge_length = 30000  # 30 km


# --- Output setup ---
output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"exec_times_cap_{qchannel_capacity}.csv")

# --- Write header ---
with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["num_nodes", "num_edges", "mean_time_s", "std_time_s"])

for nr, ne in network_sizes:
    # number of requests is proportional to network size
    num_requests = max(2, int(nr / 10))
    params = (
        f"{t_coherence}",  # Coherence time in ms (same for all memories and for all nodes)
        nr,  # Number of Routers
        ne,  # Number of Edges
        qchannel_capacity,  # Memory Capacities per qchannel
        edge_length,  # Edge length in meters
        swapping_policy,  # Swapping Order (Explicit order ['r1', 'r2', ...] Or 'ASAP')
        0.1,  # Target Fidelity
    )
    simulation_id = generate_sim_id(params)
    exec_time = simulate(simulation_id, *params, common_parameters)
    print(f"Finished ({nr}, {ne}) in {exec_time:.2f} seconds")

    # Append result
    with open(output_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([nr, ne, exec_time, 0])
