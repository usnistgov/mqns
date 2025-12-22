import configparser
import heapq
import json
import os
import random
import threading
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime

import numpy as np
from sequence.app.request_app import RequestApp
from sequence.components.memory import Memory
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.entanglement_management.swapping import (
    EntanglementSwappingA,
    EntanglementSwappingB,
    EntanglementSwappingMessage,
    SwappingMsgType,
)
from sequence.message import Message
from sequence.network_management.network_manager import ResourceReservationProtocol
from sequence.network_management.reservation import Reservation
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.topology.node import QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo

from src.utils.swaping_rules.ResourceReservationProtocol import create_rules

M = 2000
STOP_TIME = 3e12
MEMO_N = 1
MAX_WALL_TIME = 10000  # 1 hour = 3600 seconds
trs = time.time()
original_create_rules = ResourceReservationProtocol.create_rules


def _entanglement_succeed(self):
    self.memory.entangled_memory["node_id"] = self.remote_node_name
    self.memory.entangled_memory["memo_id"] = self.remote_memo_id
    self.memory.fidelity = self.memory.raw_fidelity
    self.owner.attempts_number += 1
    self.owner.success_number += 1

    self.update_resource_manager(self.memory, "ENTANGLED")


def _entanglement_fail(self):
    for event in self.scheduled_events:
        self.owner.timeline.remove_event(event)

    self.owner.attempts_number += 1
    self.owner.failed_attempts += 1

    self.update_resource_manager(self.memory, "RAW")


def emit_event(self) -> None:
    """Method to set up memory and emit photons.

    If the protocol is in round 1, the memory will be first set to the |+> state.
    Otherwise, it will apply an x_gate to the memory.
    Regardless of the round, the memory `excite` method will be invoked.

    Side Effects:
        May change state of attached memory.
        May cause attached memory to emit photon.
    """

    if self.ent_round == 1:
        self.memory.update_state(EntanglementGenerationA._plus_state)
        self.memory.generation_time = self.memory.timeline.now()
        self.memory.fidelity_time = self.memory.timeline.now()
    self.memory.excite(self.middle)
    self.owner.emit_number += 1


def start(self) -> None:
    """Method to start entanglement swapping protocol.

    Will run circuit and send measurement results to other protocols.

    Side Effects:
        Will call `update_resource_manager` method.
        Will send messages to other protocols.
    """

    assert self.left_memo.fidelity > 0
    assert self.right_memo.fidelity > 0
    assert self.left_memo.entangled_memory["node_id"] == self.left_node
    assert self.right_memo.entangled_memory["node_id"] == self.right_node

    if self.owner.get_generator().random() < self.success_probability():
        fidelity = self.updated_fidelity(self.left_memo.fidelity, self.right_memo.fidelity)
        self.is_success = True
        self.owner.succes_swapping += 1

        expire_time = min(self.left_memo.get_expire_time(), self.right_memo.get_expire_time())
        generation_time = min(self.left_memo.generation_time, self.right_memo.generation_time)
        left_fidelity_time = self.left_memo.timeline.now() - self.left_memo.fidelity_time
        right_fidelity_time = self.right_memo.timeline.now() - self.right_memo.fidelity_time

        meas_samp = self.owner.get_generator().random()
        meas_res = self.owner.timeline.quantum_manager.run_circuit(
            self.circuit, [self.left_memo.qstate_key, self.right_memo.qstate_key], meas_samp
        )
        meas_res = [meas_res[self.left_memo.qstate_key], meas_res[self.right_memo.qstate_key]]

        msg_l = EntanglementSwappingMessage(
            SwappingMsgType.SWAP_RES,
            self.left_protocol_name,
            fidelity=fidelity,
            remote_node=self.right_memo.entangled_memory["node_id"],
            remote_memo=self.right_memo.entangled_memory["memo_id"],
            expire_time=expire_time,
            meas_res=[],
            generation_time=generation_time,
            fidelity_time=right_fidelity_time,
        )
        msg_r = EntanglementSwappingMessage(
            SwappingMsgType.SWAP_RES,
            self.right_protocol_name,
            fidelity=fidelity,
            remote_node=self.left_memo.entangled_memory["node_id"],
            remote_memo=self.left_memo.entangled_memory["memo_id"],
            expire_time=expire_time,
            meas_res=meas_res,
            generation_time=generation_time,
            fidelity_time=left_fidelity_time,
        )
    else:
        msg_l = EntanglementSwappingMessage(SwappingMsgType.SWAP_RES, self.left_protocol_name, fidelity=0)
        msg_r = EntanglementSwappingMessage(SwappingMsgType.SWAP_RES, self.right_protocol_name, fidelity=0)

    self.owner.send_message(self.left_node, msg_l)
    self.owner.send_message(self.right_node, msg_r)

    self.update_resource_manager(self.left_memo, "RAW")
    self.update_resource_manager(self.right_memo, "RAW")


def memory_expire(self, memory: "Memory") -> None:
    """Method to receive expired memories.

    Args:
        memory (Memory): memory that has expired.
    """
    self.expired_memories_counter += 1
    self.resource_manager.memory_expire(memory)


def __init__(self, msg_type: SwappingMsgType, receiver: str, **kwargs):
    Message.__init__(self, msg_type, receiver)
    if self.msg_type is SwappingMsgType.SWAP_RES:
        self.fidelity = kwargs.get("fidelity")
        self.remote_node = kwargs.get("remote_node")
        self.remote_memo = kwargs.get("remote_memo")
        self.expire_time = kwargs.get("expire_time")
        self.meas_res = kwargs.get("meas_res")
        self.generation_time = kwargs.get("generation_time", 0)
        self.fidelity_time = kwargs.get("fidelity_time")
    else:
        raise Exception("Entanglement swapping protocol create unkown type of message: %s" % str(msg_type))


def received_message(self, src: str, msg: "EntanglementSwappingMessage") -> None:
    """Method to receive messages from EntanglementSwappingA.

    Args:
        src (str): name of node sending message.
        msg (EntanglementSwappingMesssage): message sent.

    Side Effects:
        Will invoke `update_resource_manager` method.
    """

    # log.logger.debug(self.owner.name + " protocol received_message from node {}, fidelity={}".format(src, msg.fidelity))

    assert src == self.remote_node_name

    if msg.fidelity > 0 and self.owner.timeline.now() < msg.expire_time:
        if msg.meas_res == [1, 0]:
            self.owner.timeline.quantum_manager.run_circuit(self.z_cir, [self.memory.qstate_key])
        elif msg.meas_res == [0, 1]:
            self.owner.timeline.quantum_manager.run_circuit(self.x_cir, [self.memory.qstate_key])
        elif msg.meas_res == [1, 1]:
            self.owner.timeline.quantum_manager.run_circuit(self.x_z_cir, [self.memory.qstate_key])

        self.memory.fidelity = msg.fidelity
        self.memory.entangled_memory["node_id"] = msg.remote_node
        self.memory.entangled_memory["memo_id"] = msg.remote_memo
        self.memory.update_expire_time(msg.expire_time)
        self.memory.generation_time = msg.generation_time
        self.memory.fidelity_time -= msg.fidelity_time
        self.update_resource_manager(self.memory, MemoryInfo.ENTANGLED)
    else:
        self.update_resource_manager(self.memory, MemoryInfo.RAW)


EntanglementGenerationA._entanglement_succeed = _entanglement_succeed
EntanglementGenerationA._entanglement_fail = _entanglement_fail
EntanglementGenerationA.emit_event = emit_event
EntanglementSwappingA.start = start
QuantumRouter.memory_expire = memory_expire

EntanglementSwappingMessage.__init__ = __init__
EntanglementSwappingB.received_message = received_message


class EnranglementRequestApp(RequestApp):
    def __init__(self, node: QuantumRouter, other_node: str):
        super().__init__(node)
        self.accumulated_fidelity = 0
        self.accumulated_age = []
        self.accumulated_fidelity_time = []
        self.other_node = other_node

    def get_memory(self, info: "MemoryInfo"):
        if info.state == "ENTANGLED" and info.remote_node == self.other_node:
            self.memory_counter += 1
            self.accumulated_fidelity += info.fidelity
            self.accumulated_age.append(self.node.timeline.now() - info.memory.generation_time)
            self.accumulated_fidelity_time.append(self.node.timeline.now() - info.memory.fidelity_time)
            self.node.resource_manager.update(None, info.memory, "RAW")

    def get_fidelity(self) -> float:
        if self.memory_counter == 0:
            return 0
        else:
            return self.accumulated_fidelity / self.memory_counter

    def get_age(self) -> float:
        if len(self.accumulated_age) == 0:
            return 0
        else:
            return sum(self.accumulated_age) / len(self.accumulated_age)

    def get_age_std(self) -> float:
        if len(self.accumulated_age) == 0:
            return 0.0
        return float(np.std(self.accumulated_age))

    def get_fidelity_time(self) -> float:
        if len(self.accumulated_fidelity_time) == 0:
            return 0.0
        return sum(self.accumulated_fidelity_time) / len(self.accumulated_fidelity_time)

    def get_fidelity_time_std(self) -> float:
        if len(self.accumulated_fidelity_time) == 0:
            return 0.0
        return float(np.std(self.accumulated_fidelity_time))

    def get_eg_probability(self) -> float:
        return self.node.success_number / self.node.attempts_number

    def get_attempts_rate(self) -> float:
        return self.node.attempts_number / (self.end_t - self.start_t) * 1e12


class ResetApp:
    def __init__(self, node, other_node_name, target_fidelity=0.1):
        self.node = node
        self.node.set_app(self)
        self.other_node_name = other_node_name
        self.target_fidelity = target_fidelity
        self.accumulated_fidelity = 0
        self.accumulated_age = []
        self.accumulated_fidelity_time = []
        self.other_node = other_node_name
        self.memory_counter = 0
        self.start_t: int = -1
        self.end_t: int = -1
        self.memory_counter: int = 0

    def set(self, start_t: int, end_t: int):
        self.start_t = start_t
        self.end_t = end_t

    def get_other_reservation(self, reservation):
        """called when receiving the request from the initiating node.

        For this application, we do not need to do anything.
        """

        pass

    def get_memory(self, info):
        """Similar to the get_memory method of the main application.

        We check if the memory info meets the request first,
        by noting the remote entangled memory and entanglement fidelity.
        We then free the memory for future use.
        """
        if info.state == "ENTANGLED" and info.remote_node == self.other_node_name and info.fidelity > self.target_fidelity:
            self.memory_counter += 1
            self.accumulated_fidelity += info.fidelity
            self.accumulated_age.append(self.node.timeline.now() - info.memory.generation_time)
            self.accumulated_fidelity_time.append(self.node.timeline.now() - info.memory.fidelity_time)
            self.node.resource_manager.update(None, info.memory, "RAW")

    def get_fidelity(self) -> float:
        if self.memory_counter == 0:
            return 0
        else:
            return self.accumulated_fidelity / self.memory_counter

    def get_age(self) -> float:
        if len(self.accumulated_age) == 0:
            return 0
        else:
            return sum(self.accumulated_age) / len(self.accumulated_age)

    def get_age_std(self) -> float:
        if len(self.accumulated_age) == 0:
            return 0.0
        return float(np.std(self.accumulated_age))

    def get_fidelity_time(self) -> float:
        if len(self.accumulated_fidelity_time) == 0:
            return 0.0
        return sum(self.accumulated_fidelity_time) / len(self.accumulated_fidelity_time)

    def get_fidelity_time_std(self) -> float:
        if len(self.accumulated_fidelity_time) == 0:
            return 0.0
        return float(np.std(self.accumulated_fidelity_time))

    def get_eg_probability(self) -> float:
        return self.node.success_number / self.node.attempts_number

    def get_attempts_rate(self) -> float:
        return self.node.attempts_number / (self.end_t - self.start_t) * 1e12

    def get_throughput(self) -> float:
        return self.memory_counter / (self.end_t - self.start_t) * 1e12


# Context manager to temporarily override shared attributes
@contextmanager
def override_reservation(link_capacity, swapping_order):
    # Save original values
    original_link_capacity = deepcopy(Reservation.link_capacity)
    original_create_rules = ResourceReservationProtocol.create_rules
    original_swapping_order = deepcopy(Reservation.swapping_order)

    try:
        # Override for the current simulation
        Reservation.link_capacity = link_capacity
        Reservation.swapping_order = swapping_order
        ResourceReservationProtocol.create_rules = create_rules
        yield
    finally:
        # Restore original values
        Reservation.link_capacity = original_link_capacity
        Reservation.swapping_order = original_swapping_order
        ResourceReservationProtocol.create_rules = original_create_rules


def list_to_swapping_order(swap_tree):
    result = []
    for item in swap_tree:
        if isinstance(item, int):
            result.append("r" + str(item))
        else:
            for element in item:
                result.append("r" + str(element))
    return result


def dijkstra_hop_distances(adj):
    """Compute shortest path lengths (in hops) between all pairs."""
    n = len(adj)
    dist = [[float("inf")] * n for _ in range(n)]
    for src in range(n):
        dist[src][src] = 0
        heap = [(0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[src][u]:
                continue
            for v in range(n):
                if adj[u][v]:
                    if dist[src][v] > d + 1:
                        dist[src][v] = d + 1
                        heapq.heappush(heap, (dist[src][v], v))
    return dist


def create_random_quantum_network(num_nodes, num_edges, edge_length, output_file, attenuation=0.0002):
    """
    Create a random connected topology compatible with SeQUeNCe, matching MQNS.RandomTopology.

    Args:
        num_nodes (int): total number of quantum routers
        avg_degree (float): desired average node degree (e.g., 2.5)
        edge_length (float): average physical span of the network (used for scaling link distances)
        output_file (str): JSON path to save topology
        attenuation (float): attenuation coefficient (default 0.0002)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # -----------------------------------
    # 1. Initialize node names
    # -----------------------------------
    nodes = [f"r{i}" for i in range(num_nodes)]
    N = len(nodes)

    target_edges = num_edges

    # adjacency matrix
    adj = [[0] * N for _ in range(N)]
    qconnections = []

    # -----------------------------------
    # 2. Build a random spanning tree to ensure connectivity
    # -----------------------------------
    for i in range(1, N):
        j = random.randint(0, i - 1)
        adj[i][j] = adj[j][i] = 1
        qconnections.append(
            {
                "node1": nodes[i],
                "node2": nodes[j],
                "distance": edge_length,
                "attenuation": attenuation,
                "type": "meet_in_the_middle",
            }
        )

    # -----------------------------------
    # 3. Add random extra edges until we reach target_edges
    # -----------------------------------
    while len(qconnections) < target_edges:
        a, b = random.sample(range(N), 2)
        if adj[a][b] == 0:
            adj[a][b] = adj[b][a] = 1
            qconnections.append(
                {
                    "node1": nodes[a],
                    "node2": nodes[b],
                    "distance": edge_length,
                    "attenuation": attenuation,
                    "type": "meet_in_the_middle",
                }
            )

    # -----------------------------------
    # 4. Classical connections mirror quantum ones
    # -----------------------------------
    # --- Compute hop-based shortest paths ---
    hop_distances = dijkstra_hop_distances(adj)

    # --- Classical channels: full mesh ---
    cconnections = []
    for i in range(N):
        for j in range(i + 1, N):
            hops = hop_distances[i][j]
            # Reflect hop distance as length * hops
            cconnections.append(
                {
                    "node1": nodes[i],
                    "node2": nodes[j],
                    "distance": hops * edge_length,  # classical distance ~ hop count × quantum edge length
                }
            )

    # -----------------------------------
    # 5. Build the network JSON
    # -----------------------------------
    network = {
        "nodes": [{"name": name, "type": "QuantumRouter", "seed": 0, "memo_size": M} for name in nodes],
        "qconnections": qconnections,
        "cconnections": cconnections,
        "is_parallel": False,
        "stop_time": STOP_TIME,
    }

    with open(output_file, "w") as f:
        json.dump(network, f, indent=2)

    return network


def set_parameters(topology: RouterNetTopo, config):
    # Print all sections and their keys
    """for section in config.sections():
    print(f"[{section}]")
    for key, value in config[section].items():
        print(f"{key} = {value}")
    print()"""
    # Get memory parameters
    MEMO_EXPIRE = float(config.get("Memory", "coherence_time"))
    MEMO_EFFICIENCY = float(config.get("Memory", "efficiency"))
    MEMO_FIDELITY = float(config.get("Memory", "fidelity"))
    WAVE_LENGTH = float(config.get("Memory", "wavelength"))

    # Set memory parameters
    for node in topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        memory_array = node.get_components_by_type("MemoryArray")[0]
        memory_array.update_memory_params("coherence_time", MEMO_EXPIRE)
        memory_array.update_memory_params("efficiency", MEMO_EFFICIENCY)
        memory_array.update_memory_params("raw_fidelity", MEMO_FIDELITY)
        memory_array.update_memory_params("wavelength", WAVE_LENGTH)

    # Get detector parameters
    DETECTOR_EFFICIENCY = float(config.get("Detector", "efficiency"))
    DETECTOR_COUNTRATE = float(config.get("Detector", "count_rate"))

    # Set detector parameters
    for node in topology.get_nodes_by_type(RouterNetTopo.BSM_NODE):
        bsm = node.get_components_by_type("SingleAtomBSM")[0]
        bsm.update_detectors_params("efficiency", DETECTOR_EFFICIENCY)
        bsm.update_detectors_params("count_rate", DETECTOR_COUNTRATE)

    # Get entanglement swapping parameters
    SWAPPING_SUCCESS_RATE = float(config.get("Swapping", "success_rate"))

    # Set entanglement swapping parameters
    for node in topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        node.network_manager.protocol_stack[1].set_swapping_success_rate(SWAPPING_SUCCESS_RATE)


def simulate(
    simulation_id,
    coherence_time,
    number_of_routers,
    num_edges,
    memory_capacity,
    edge_length,
    swapping_strategy,
    target_fidelity,
    config=None,
):
    """
    Simulates the quantum network with delays to test parallel execution.

    Returns:
        dict: Simulated data.
    """
    if not config:
        config_path = os.path.join(os.path.dirname(__file__), "..", "data", "configs", "common_parameters.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

    Reservation.swapping_order = None
    Reservation.link_capacity = None
    # Use context manager to encapsulate shared state changes
    with override_reservation(memory_capacity, swapping_strategy):
        topology_file = f"data/topology/topo_{simulation_id}.json"

        create_random_quantum_network(
            num_nodes=number_of_routers, num_edges=num_edges, edge_length=edge_length, output_file=topology_file
        )
        # simulation_result  = simulate(topology_file, config, swapping_order=swp_order, GUI = GUI)
        network_topo = RouterNetTopo(topology_file)

        config.set("Memory", "coherence_time", coherence_time)
        set_parameters(network_topo, config)

        tl = network_topo.get_timeline()
        tl.stop_time = STOP_TIME
        tl.show_progress = False
        # get quantum routers
        routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
        router_names = [r.name for r in routers]
        for r in routers:
            r.network_manager.network_routers = routers
        # choose number of random S–D pairs (≈ 20% of nodes)
        num_requests = max(2, int(len(router_names) / 10))
        print(f"Generating {num_requests} random requests...")
        # ensure no node participates in more than one request
        available_nodes = router_names.copy()
        random.shuffle(available_nodes)
        # keep track of applications
        apps = []
        for _ in range(num_requests):
            if len(available_nodes) < 2:
                print("Not enough nodes left to create more unique src-dst pairs.")
                break

            # take two distinct unused nodes
            src_name = available_nodes.pop()
            dst_name = available_nodes.pop()

            src_node = next(r for r in routers if r.name == src_name)
            dst_node = next(r for r in routers if r.name == dst_name)

            app_src = EnranglementRequestApp(src_node, dst_name)
            app_dst = ResetApp(dst_node, src_name)

            apps.append((app_src, app_dst))
        for r in routers:
            r.success_number = r.attempts_number = r.failed_attempts = 0
            r.emit_number = r.succes_swapping = r.expired_memories_counter = 0
        # start simulation
        sim_start_time = datetime.now()
        tl.init()

        start_time = 0.1e12
        end_time = STOP_TIME
        memory_number = 1
        # start all requests
        for app_src, app_dst in apps:
            app_src.start(app_dst.node.name, start_time, end_time, memory_number, target_fidelity)
            app_dst.set(start_time, end_time)
        stop_event = threading.Event()

        def stop_timeline_after_timeout():
            # Wait up to MAX_WALL_TIME or until stop_event is set
            if not stop_event.wait(timeout=MAX_WALL_TIME):
                print(f"Simulation exceeded {MAX_WALL_TIME} seconds — stopping timeline at {tl.time / 1e12}.")
                tl.stop()
            else:
                # Simulation ended before timeout
                print(f"Stopper thread exited early (simulation finished). stopping timeline at {tl.time / 1e12}.")

        stopper = threading.Thread(target=stop_timeline_after_timeout, daemon=True)
        stopper.start()
        try:
            # run the timeline
            tl.run()
        except Exception:
            print("An error occurred during simulation:")
            traceback.print_exc()
        # Notify the stopper thread to exit immediately
        stop_event.set()

    sim_end_time = datetime.now()
    simulation_duration = (sim_end_time - sim_start_time).total_seconds()
    if os.path.exists(topology_file):
        os.remove(topology_file)  # Deletes the file
        print(f"File '{topology_file}' has been deleted.")
    else:
        print(f"File '{topology_file}' does not exist.")

    return simulation_duration
