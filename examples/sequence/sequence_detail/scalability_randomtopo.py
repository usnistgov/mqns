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
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.topology.node import QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo


def _EntanglementGenerationA_entanglement_succeed(self: EntanglementGenerationA):
    self.memory.entangled_memory["node_id"] = self.remote_node_name
    self.memory.entangled_memory["memo_id"] = self.remote_memo_id
    self.memory.fidelity = self.memory.raw_fidelity
    self.owner.attempts_number += 1
    self.owner.success_number += 1

    self.update_resource_manager(self.memory, "ENTANGLED")


def _EntanglementGenerationA_entanglement_fail(self: EntanglementGenerationA):
    for event in self.scheduled_events:
        self.owner.timeline.remove_event(event)

    self.owner.attempts_number += 1
    self.owner.failed_attempts += 1

    self.update_resource_manager(self.memory, "RAW")


def _EntanglementGenerationA_emit_event(self: EntanglementGenerationA) -> None:
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


def _EntanglementSwappingA_start(self: EntanglementSwappingA) -> None:
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
        self.owner.success_swapping += 1

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


def _QuantumRouter_memory_expire(self: QuantumRouter, memory: "Memory") -> None:
    """Method to receive expired memories.

    Args:
        memory (Memory): memory that has expired.
    """
    self.expired_memories_counter += 1
    self.resource_manager.memory_expire(memory)


def _EntanglementSwappingMessage_init(self: EntanglementSwappingMessage, msg_type: SwappingMsgType, receiver: str, **kwargs):
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
        raise Exception("Entanglement swapping protocol create unknown type of message: %s" % str(msg_type))


def _EntanglementSwappingB_received_message(self: EntanglementSwappingB, src: str, msg: "EntanglementSwappingMessage") -> None:
    """Method to receive messages from EntanglementSwappingA.

    Args:
        src (str): name of node sending message.
        msg (EntanglementSwappingMesssage): message sent.

    Side Effects:
        Will invoke `update_resource_manager` method.
    """

    if src != self.remote_node_name:
        return

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


EntanglementGenerationA._entanglement_succeed = _EntanglementGenerationA_entanglement_succeed
EntanglementGenerationA._entanglement_fail = _EntanglementGenerationA_entanglement_fail
EntanglementGenerationA.emit_event = _EntanglementGenerationA_emit_event
EntanglementSwappingA.start = _EntanglementSwappingA_start
QuantumRouter.memory_expire = _QuantumRouter_memory_expire
EntanglementSwappingMessage.__init__ = _EntanglementSwappingMessage_init
EntanglementSwappingB.received_message = _EntanglementSwappingB_received_message


class EntanglementRequestApp(RequestApp):
    def __init__(self, node: QuantumRouter, other_node: str):
        super().__init__(node)
        self.other_node = other_node

    def get_memory(self, info: "MemoryInfo"):
        if info.state == "ENTANGLED" and info.remote_node == self.other_node:
            self.memory_counter += 1
            self.node.resource_manager.update(None, info.memory, "RAW")


class ResetApp:
    def __init__(self, node: QuantumRouter, other_node_name: str, target_fidelity=0.1):
        self.node = node
        self.node.set_app(self)
        self.other_node_name = other_node_name
        self.target_fidelity = target_fidelity
        self.start_t: int = -1
        self.end_t: int = -1

    def set(self, start_t: int, end_t: int):
        self.start_t = start_t
        self.end_t = end_t

    def get_other_reservation(self, reservation):
        """called when receiving the request from the initiating node.

        For this application, we do not need to do anything.
        """

        pass

    def get_memory(self, info: "MemoryInfo"):
        """Similar to the get_memory method of the main application.

        We check if the memory info meets the request first,
        by noting the remote entangled memory and entanglement fidelity.
        We then free the memory for future use.
        """
        if info.state == "ENTANGLED" and info.remote_node == self.other_node_name and info.fidelity > self.target_fidelity:
            self.node.resource_manager.update(None, info.memory, "RAW")


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
