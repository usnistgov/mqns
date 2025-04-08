import logging

from qns.network.route.dijkstra import DijkstraRouteAlgorithm
from qns.simulator.simulator import Simulator
from qns.network import QuantumNetwork, TimingModeEnum
import qns.utils.log as log
from qns.utils.rnd import set_seed
from qns.network.protocol.proactive_routing import ProactiveRouting
from qns.network.protocol.link_layer import LinkLayer
from qns.network.protocol.proactive_routing_controller import ProactiveRoutingControllerApp
from qns.network.topology.customtopo import CustomTopology
import numpy as np

from qns.entity.monitor import Monitor
from qns.entity.qchannel import RecvQubitPacket
from qns.entity.cchannel import RecvClassicPacket


log.logger.setLevel(logging.DEBUG)


light_speed = 2 * 10**5 # km/s

def drop_rate(length):
    # drop 0.2 db/KM
    return 10 ** (- (0.2 * length) / 10)

# constrains
init_fidelity = 0.99
t_coherence = 10    # sec

# set a fixed random seed
set_seed(150)
s = Simulator(0, 2 + 5e-06, accuracy=1000000)
log.install(s)

ch_1 = 5
ch_2 = 5
ch_3 = 5
t_slot = 2*(ch_1 + ch_2 + ch_3) / light_speed

topo_3_nodes = {
    "qnodes": [
        {
            "name": "S",
            "memory": {
                "decoherence_rate": 1 / t_coherence,
                # Coherence time T = 1 / decoherence_rate = 0.1s  (100ms)
                # e.g., a qubit is considered effectively decohered when coherence drops below 1% or 0.1%, which happens at t ≈ 4.6T to 6.9T.
            },
            "apps": [LinkLayer(attempt_rate=1000, init_fidelity=init_fidelity), ProactiveRouting()]
        },
        {
            "name": "R1",
            "memory": {
                "decoherence_rate": 1 / t_coherence,
            },
            "apps": [LinkLayer(attempt_rate=1000, init_fidelity=init_fidelity), ProactiveRouting()]
        },
        {
            "name": "R2",
            "memory": {
                "decoherence_rate": 1 / t_coherence,
            },
            "apps": [LinkLayer(attempt_rate=1000, init_fidelity=init_fidelity), ProactiveRouting()]
        },
        {
            "name": "D",
            "memory": {
                "decoherence_rate": 1 / t_coherence,
            },
            "apps": [LinkLayer(attempt_rate=1000, init_fidelity=init_fidelity), ProactiveRouting()]
        }
    ],
    "qchannels": [
        { "node1": "S", "node2":"R1", "capacity": 1, "parameters": {"length": ch_1, "delay": ch_1 / light_speed, "drop_rate": drop_rate(ch_1)} },
        { "node1": "R1", "node2":"R2", "capacity": 1, "parameters": {"length": ch_2, "delay": ch_2 / light_speed, "drop_rate": drop_rate(ch_2)} },
        { "node1": "R2", "node2":"D", "capacity": 1, "parameters": {"length": ch_3, "delay": ch_3 / light_speed, "drop_rate": drop_rate(ch_3)} }
    ],
    "cchannels": [
        { "node1": "S", "node2":"R1", "parameters": {"length": ch_1, "delay": ch_1 / light_speed} },
        { "node1": "R1", "node2":"R2", "parameters": {"length": ch_2, "delay": ch_2 / light_speed} },
        { "node1": "R2", "node2":"D", "parameters": {"length": ch_3, "delay": ch_3 / light_speed} },
        { "node1": "ctrl", "node2":"S", "parameters": {"length": 1.0, "delay": 1 / light_speed} },
        { "node1": "ctrl", "node2":"R1", "parameters": {"length": 1.0, "delay": 1 / light_speed} },
        { "node1": "ctrl", "node2":"R2", "parameters": {"length": 1.0, "delay": 1 / light_speed} },
        { "node1": "ctrl", "node2":"D", "parameters": {"length": 1.0, "delay": 1 / light_speed} }
    ],
    "controller": {
        "name": "ctrl",
        "apps": [ProactiveRoutingControllerApp()]
    }
}

topo = CustomTopology(topo_3_nodes)

# controller is set at the QuantumNetwork object, so we can use existing topologies and their builders
net = QuantumNetwork(topo=topo, route=DijkstraRouteAlgorithm(), timing_mode=TimingModeEnum.LSYNC, t_slot=t_slot)

# net.build_route()
# net.random_requests(requests_number, attr={"send_rate": send_rate})

capacity_counts = {}
def watch_send_count(simulator, network, event):
    if event.qchannel.name in capacity_counts:
        capacity_counts[event.qchannel.name]+=1
    else:
        capacity_counts[event.qchannel.name] = 1

    return event.qchannel.name

swap_count = { "R1": 0, "R2": 0 }
def watch_swap_count(simulator, network, event):
    if event.packet.get()["cmd"] == "SWAP_UPDATE":
        if not event.packet.get()['fwd']:
            swap_count[event.packet.get()['swapping_node']]+=1
    return swap_count

monitor1 = Monitor(name="monitor_1", network = None)
monitor1.add_attribution(name="send_count", calculate_func=watch_send_count)
monitor1.at_event(RecvQubitPacket)

monitor2 = Monitor(name="monitor_2", network = None)
monitor2.add_attribution(name="swap_count", calculate_func=watch_swap_count)
monitor2.at_event(RecvClassicPacket)

net.install(s)

monitor1.install(s)
monitor2.install(s)

s.run()
#data = monitor2.get_data()
#print(data)

print(capacity_counts)
[print(f"{k}: {v/2}") for k, v in swap_count.items()]




# s.run_continuous()

# import signal
# def stop_emulation(sig, frame):
#     print('Stopping simulation...')
#     s.stop()
# signal.signal(signal.SIGINT, stop_emulation)

#results = []
#for req in net.requests:
#    src = req.src
#    results.append(src.apps[0].success_count)
#fair = sum(results)**2 / (len(results) * sum([r**2 for r in results]))
#log.monitor(requests_number, nodes_number, s.time_spend, sep=" ")
