from typing import Any, Dict, List

from sequence.entanglement_management.entanglement_protocol import EntanglementProtocol
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.entanglement_management.purification import BBPSSW
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.resource_management.memory_manager import MemoryInfo, MemoryManager
from sequence.resource_management.resource_manager import Reservation
from sequence.resource_management.rule_manager import Arguments, Rule
from sequence.topology.node import QuantumRouter


def eg_rule_condition(memory_info: "MemoryInfo", manager: "MemoryManager", args: Arguments) -> List["MemoryInfo"]:
    """Condition function used by entanglement generation protocol on nodes"""
    memory_indices = args["memory_indices"]
    if memory_info.state == "RAW" and memory_info.index in memory_indices:
        return [memory_info]
    else:
        return []


def eg_rule_action1(memories_info: List["MemoryInfo"], args: Dict[str, Any]):
    """Action function used by entanglement generation protocol on nodes except
    the initiator

    """
    memories = [info.memory for info in memories_info]
    memory = memories[0]
    mid = args["mid"]
    path = args["path"]
    index = args["index"]
    protocol = EntanglementGenerationA(None, "EGA." + memory.name, mid, path[index - 1], memory)
    return protocol, [None], [None], [None]


def eg_req_func(protocols: List["EntanglementProtocol"], args: Arguments) -> "EntanglementGenerationA":
    """Function used by `eg_rule_action2` function for selecting generation
    protocols on the remote node

    """
    name = args["name"]
    reservation = args["reservation"]
    for protocol in protocols:
        if (
            isinstance(protocol, EntanglementGenerationA)
            and protocol.remote_node_name == name
            and protocol.rule.get_reservation() == reservation
        ):
            return protocol


def eg_rule_action2(memories_info: List["MemoryInfo"], args: Arguments):
    """Action function used by entanglement generation protocol on nodes except
    the responder

    """
    mid = args["mid"]
    path = args["path"]
    index = args["index"]
    memories = [info.memory for info in memories_info]
    memory = memories[0]
    protocol = EntanglementGenerationA(None, "EGA." + memory.name, mid, path[index + 1], memory)
    req_args = {"name": args["name"], "reservation": args["reservation"]}
    return protocol, [path[index + 1]], [eg_req_func], [req_args]


def ep_rule_condition1(memory_info: "MemoryInfo", manager: "MemoryManager", args: Arguments):
    """Condition function used by BBPSSW protocol on nodes except the initiator"""
    memory_indices = args["memory_indices"]
    reservation = args["reservation"]
    if memory_info.index in memory_indices and memory_info.state == "ENTANGLED" and memory_info.fidelity < reservation.fidelity:
        for info in manager:
            if (
                info != memory_info
                and info.index in memory_indices
                and info.state == "ENTANGLED"
                and info.remote_node == memory_info.remote_node
                and info.fidelity == memory_info.fidelity
            ):
                assert memory_info.remote_memo != info.remote_memo
                return [memory_info, info]
    return []


def ep_req_func1(protocols, args: Arguments) -> "BBPSSW":
    """Function used by `ep_rule_action1` for selecting purification protocols
    on the remote node

    """
    remote0 = args["remote0"]
    remote1 = args["remote1"]

    _protocols = []
    for protocol in protocols:
        if not isinstance(protocol, BBPSSW):
            continue

        if protocol.kept_memo.name == remote0:
            _protocols.insert(0, protocol)
        if protocol.kept_memo.name == remote1:
            _protocols.insert(1, protocol)

    if len(_protocols) != 2:
        return None

    protocols.remove(_protocols[1])
    _protocols[1].rule.protocols.remove(_protocols[1])
    _protocols[1].kept_memo.detach(_protocols[1])
    _protocols[0].meas_memo = _protocols[1].kept_memo
    _protocols[0].memories = [_protocols[0].kept_memo, _protocols[0].meas_memo]
    _protocols[0].name = _protocols[0].name + "." + _protocols[0].meas_memo.name
    _protocols[0].meas_memo.attach(_protocols[0])

    return _protocols[0]


def ep_rule_action1(memories_info: List["MemoryInfo"], args: Arguments):
    """Action function used by BBPSSW protocol on nodes except the
    responder node

    """
    memories = [info.memory for info in memories_info]
    name = "EP.%s.%s" % (memories[0].name, memories[1].name)
    protocol = BBPSSW(None, name, memories[0], memories[1])
    dsts = [memories_info[0].remote_node]
    req_funcs = [ep_req_func1]
    req_args = [
        {"remote0": memories_info[0].remote_memo, "remote1": memories_info[1].remote_memo},
    ]
    return protocol, dsts, req_funcs, req_args


def ep_rule_condition2(memory_info: "MemoryInfo", manager: "MemoryManager", args: Arguments) -> List["MemoryInfo"]:
    """Condition function used by BBPSSW protocol on nodes except the responder"""
    memory_indices = args["memory_indices"]
    fidelity = args["fidelity"]

    if memory_info.index in memory_indices and memory_info.state == "ENTANGLED" and memory_info.fidelity < fidelity:
        return [memory_info]
    return []


def ep_rule_action2(memories_info: List["MemoryInfo"], args: Arguments):
    """Action function used by BBPSSW protocol on nodes except the responder"""
    memories = [info.memory for info in memories_info]
    name = "EP.%s" % memories[0].name
    protocol = BBPSSW(None, name, memories[0], None)
    return protocol, [None], [None], [None]


def es_rule_actionB(memories_info: List["MemoryInfo"], args):
    memories = [info.memory for info in memories_info]
    memory = memories[0]
    protocol = EntanglementSwappingB(None, "ESB." + memory.name, memory)
    return [protocol, [None], [None], [None]]


def es_rule_ASAPactionB(memories_info: List["MemoryInfo"], args: Arguments):
    """Action function used by EntanglementSwappingB protocol"""
    memories = [info.memory for info in memories_info]
    memory = memories[0]
    protocol = EntanglementSwappingB(None, "ESB." + memory.name, memory)
    #### expire the rule
    node = args["node"]
    rule = args["rule"]
    # memories_info[0].to_entangled()
    node.resource_manager.rule_manager.expire(rule=rule)
    return protocol, [None], [None], [None]


def es_rule_conditionB1(memory_info: "MemoryInfo", manager: "MemoryManager", args: Arguments):
    """Condition function used by EntanglementSwappingB protocol on nodes of either responder or initiator"""
    memory_indices = args["memory_indices"]
    target_remote = args["target_remote"]
    fidelity = args["fidelity"]
    if (
        memory_info.state == "ENTANGLED"
        and memory_info.index in memory_indices
        # and memory_info.remote_node != path[-1]
        and memory_info.remote_node != target_remote
        # and memory_info.fidelity >= reservation.fidelity):
        and memory_info.fidelity >= fidelity
    ):
        return [memory_info]
    else:
        return []


def es_rule_ASAP_conditionB1(memory_info: "MemoryInfo", manager: "MemoryManager", args: Arguments):
    """Condition function used by EntanglementSwappingB protocol on nodes of either responder or initiator"""
    desired_memory = args["desired_memory"]
    # if memory_info.state == "RAW": delattr(memory_info, 'lock')
    # print("will verify B1 conditions for ", memory_info.memory.name, memory_info.state, target_remote_memory.name)
    if memory_info.state == "ENTANGLED" and memory_info.memory.name == desired_memory:
        if hasattr(memory_info, "lock"):
            delattr(memory_info, "lock")
            return [memory_info]
        else:
            return []
    else:
        return []


def es_rule_conditionA(memory_info: "MemoryInfo", manager: "MemoryManager", args: Arguments):
    """Condition function used by EntanglementSwappingA protocol on nodes"""
    memory_indices = args["memory_indices"]
    left = args["left"]
    right = args["right"]
    fidelity = args["fidelity"]
    if (
        memory_info.state == "ENTANGLED"
        and memory_info.index in memory_indices
        and memory_info.remote_node == left
        and memory_info.fidelity >= fidelity
    ):
        for info in manager:
            if (
                info.state == "ENTANGLED"
                and info.index in memory_indices
                and info.remote_node == right
                and info.fidelity >= fidelity
            ):
                return [memory_info, info]
    elif (
        memory_info.state == "ENTANGLED"
        and memory_info.index in memory_indices
        and memory_info.remote_node == right
        and memory_info.fidelity >= fidelity
    ):
        for info in manager:
            if (
                info.state == "ENTANGLED"
                and info.index in memory_indices
                and info.remote_node == left
                and info.fidelity >= fidelity
            ):
                return [memory_info, info]
    return []


def load_B_rules(left_node: "QuantumRouter", left_node_memory: "str", right_node: "QuantumRouter", right_node_memory: "str"):
    condition_args = {"desired_memory": left_node_memory}
    action_args = {}
    rule = Rule(5, es_rule_ASAPactionB, es_rule_ASAP_conditionB1, action_args, condition_args)
    rule.action_args = {"node": left_node, "rule": rule}
    left_memory_info = next(
        (m for m in left_node.resource_manager.memory_manager.memory_map if m.memory.name == left_node_memory), None
    )
    left_memory_info.lock = True
    left_node.resource_manager.load(rule=rule)

    # left_node.resource_manager.rule_manager.expire(rule=rule)

    condition_args = {"desired_memory": right_node_memory}
    action_args = {}
    rule = Rule(5, es_rule_ASAPactionB, es_rule_ASAP_conditionB1, action_args, condition_args)
    rule.action_args = {"node": right_node, "rule": rule}
    right_memory_info = next(
        (m for m in right_node.resource_manager.memory_manager.memory_map if m.memory.name == right_node_memory), None
    )
    right_memory_info.lock = True
    right_node.resource_manager.load(rule=rule)
    # right_node.resource_manager.rule_manager.expire(rule=rule)


def loadB(left_node: "QuantumRouter", left_node_memory: "str", right_node: "QuantumRouter", right_node_memory: "str"):
    for memo in left_node.get_components_by_type("MemoryArray")[0]:
        # print(memo.name,left_node_memory)
        if memo.name == left_node_memory:
            protocol = EntanglementSwappingB(None, "ESB." + left_node_memory, memo)
            left_node.resource_manager.update(protocol=protocol, memory=memo, state="OCCUPIED")
    for memo in right_node.get_components_by_type("MemoryArray")[0]:
        if memo.name == right_node_memory:
            protocol = EntanglementSwappingB(None, "ESB." + right_node_memory, memo)
            right_node.resource_manager.update(protocol=protocol, memory=memo, state="OCCUPIED")


def es_rule_ASAP_conditionA(memory_info: "MemoryInfo", manager: "MemoryManager", args: Arguments):
    """Condition function used by EntanglementSwappingA protocol on nodes"""
    memory_indices = args["memory_indices"]
    fidelity = args["fidelity"]
    path = args["path"]
    index = args["index"]
    routers_path = manager.resource_manager.owner.network_manager.network_routers
    if hasattr(memory_info, "lock"):
        return []
    if (
        memory_info.state == "ENTANGLED"
        and memory_info.index in memory_indices
        and memory_info.remote_node in path[:index]
        and memory_info.fidelity >= fidelity
    ):
        for info in manager:
            if (
                info.state == "ENTANGLED"
                and info.index in memory_indices
                and info.remote_node in path[index + 1 :]
                and info.fidelity >= fidelity
            ):
                # add to apply A

                load_B_rules(
                    left_node=next((r for r in routers_path if r.name == memory_info.remote_node), None),
                    left_node_memory=memory_info.remote_memo,
                    right_node=next((r for r in routers_path if r.name == info.remote_node), None),
                    right_node_memory=info.remote_memo,
                )
                return [memory_info, info]
    elif (
        memory_info.state == "ENTANGLED"
        and memory_info.index in memory_indices
        and memory_info.remote_node in path[index + 1 :]
        and memory_info.fidelity >= fidelity
    ):
        for info in manager:
            if (
                info.state == "ENTANGLED"
                and info.index in memory_indices
                and info.remote_node in path[:index]
                and info.fidelity >= fidelity
            ):
                # add logic to apply A
                load_B_rules(
                    left_node=next((r for r in routers_path if r.name == info.remote_node), None),
                    left_node_memory=info.remote_memo,
                    right_node=next((r for r in routers_path if r.name == memory_info.remote_node), None),
                    right_node_memory=memory_info.remote_memo,
                )
                return [memory_info, info]
    return []


def es_req_func(protocols: List["EntanglementProtocol"], args: Arguments) -> "EntanglementSwappingB":
    """Function used by `es_rule_actionA` for selecting swapping protocols on the remote node"""
    target_memo = args["target_memo"]
    for protocol in protocols:
        if (
            isinstance(protocol, EntanglementSwappingB)
            # and protocol.memory.name == memories_info[0].remote_memo):
            and protocol.memory.name == target_memo
        ):
            return protocol


def es_rule_actionA(memories_info: List["MemoryInfo"], args: Arguments):
    """Action function used by EntanglementSwappingA protocol on nodes"""
    es_succ_prob = args["es_succ_prob"]
    es_degradation = args["es_degradation"]
    memories = [info.memory for info in memories_info]
    protocol = EntanglementSwappingA(
        None,
        "ESA.%s.%s" % (memories[0].name, memories[1].name),
        memories[0],
        memories[1],
        success_prob=es_succ_prob,
        degradation=es_degradation,
    )
    dsts = [info.remote_node for info in memories_info]
    req_funcs = [es_req_func, es_req_func]
    req_args = [{"target_memo": memories_info[0].remote_memo}, {"target_memo": memories_info[1].remote_memo}]
    return protocol, dsts, req_funcs, req_args


def es_rule_conditionB2(memory_info: "MemoryInfo", manager: "MemoryManager", args: Arguments) -> List["MemoryInfo"]:
    """Condition function used by EntanglementSwappingB protocol on intermediate nodes of path"""
    memory_indices = args["memory_indices"]
    left = args["left"]
    right = args["right"]
    fidelity = args["fidelity"]
    if (
        memory_info.state == "ENTANGLED"
        and memory_info.index in memory_indices
        and memory_info.remote_node not in [left, right]
        and memory_info.fidelity >= fidelity
    ):
        return [memory_info]
    else:
        return []


def es_rule_ASAP_conditionB2(memory_info: "MemoryInfo", manager: "MemoryManager", args: Arguments) -> List["MemoryInfo"]:
    """Condition function used by EntanglementSwappingB protocol on intermediate nodes of path"""
    memory_indices = args["memory_indices"]
    fidelity = args["fidelity"]
    if memory_info.state == "ENTANGLED":
        print("B ", memory_info.state)
    if memory_info.state == "ENTANGLED" and memory_info.index in memory_indices and memory_info.fidelity >= fidelity:
        return [memory_info]
    else:
        return []


def create_rules(self, path: List[str], reservation: "Reservation") -> List["Rule"]:
    """Method to create rules for a successful request.

    Rules are used to direct the flow of information/entanglement in the resource manager.

    Args:
        path (List[str]): list of node names in entanglement path.
        reservation (Reservation): approved reservation.

    Returns:
        List[Rule]: list of rules created by the method.
    """
    rules = []
    memory_indices = []
    for card in self.timecards:
        memory_indices.append(card.memory_index)

    # create rules for entanglement generation
    index = path.index(self.owner.name)
    if reservation.link_capacity:
        if index > 0:
            condition_args = {"memory_indices": memory_indices[: reservation.link_capacity]}
            action_args = {"mid": self.owner.map_to_middle_node[path[index - 1]], "path": path, "index": index}
            rule = Rule(10, eg_rule_action1, eg_rule_condition, action_args, condition_args)
            rules.append(rule)
        if index < len(path) - 1:
            if index == 0:
                condition_args = {"memory_indices": memory_indices[: reservation.link_capacity]}
            else:
                condition_args = {
                    "memory_indices": memory_indices[
                        reservation.link_capacity : reservation.link_capacity + reservation.link_capacity
                    ]
                }

            action_args = {
                "mid": self.owner.map_to_middle_node[path[index + 1]],
                "path": path,
                "index": index,
                "name": self.owner.name,
                "reservation": reservation,
            }
            rule = Rule(10, eg_rule_action2, eg_rule_condition, action_args, condition_args)
            rules.append(rule)

        # create rules for entanglement purification
        if index > 0:
            condition_args = {"memory_indices": memory_indices[: reservation.link_capacity], "reservation": reservation}
            action_args = {}
            rule = Rule(10, ep_rule_action1, ep_rule_condition1, action_args, condition_args)
            rules.append(rule)

        if index < len(path) - 1:
            if index == 0:
                condition_args = {
                    "memory_indices": memory_indices[: reservation.link_capacity],
                    "fidelity": reservation.fidelity,
                }
            else:
                condition_args = {
                    "memory_indices": memory_indices[
                        reservation.link_capacity : reservation.link_capacity + reservation.link_capacity
                    ],
                    "fidelity": reservation.fidelity,
                }

            action_args = {}
            rule = Rule(10, ep_rule_action2, ep_rule_condition2, action_args, condition_args)
            rules.append(rule)

    else:
        if index > 0:
            condition_args = {"memory_indices": memory_indices[: reservation.memory_size]}
            action_args = {"mid": self.owner.map_to_middle_node[path[index - 1]], "path": path, "index": index}
            rule = Rule(10, eg_rule_action1, eg_rule_condition, action_args, condition_args)
            rules.append(rule)

        if index < len(path) - 1:
            if index == 0:
                condition_args = {"memory_indices": memory_indices[: reservation.memory_size]}
            else:
                condition_args = {"memory_indices": memory_indices[reservation.memory_size :]}

            action_args = {
                "mid": self.owner.map_to_middle_node[path[index + 1]],
                "path": path,
                "index": index,
                "name": self.owner.name,
                "reservation": reservation,
            }
            rule = Rule(10, eg_rule_action2, eg_rule_condition, action_args, condition_args)
            rules.append(rule)

        # create rules for entanglement purification
        if index > 0:
            condition_args = {"memory_indices": memory_indices[: reservation.memory_size], "reservation": reservation}
            action_args = {}
            rule = Rule(10, ep_rule_action1, ep_rule_condition1, action_args, condition_args)
            rules.append(rule)

        if index < len(path) - 1:
            if index == 0:
                condition_args = {"memory_indices": memory_indices, "fidelity": reservation.fidelity}
            else:
                condition_args = {"memory_indices": memory_indices[reservation.memory_size :], "fidelity": reservation.fidelity}

            action_args = {}
            rule = Rule(10, ep_rule_action2, ep_rule_condition2, action_args, condition_args)
            rules.append(rule)

    # create rules for entanglement swapping

    if reservation.swapping_order == "ASAP":
        condition_args = {"memory_indices": memory_indices, "fidelity": reservation.fidelity, "index": index, "path": path}
        action_args = {"es_succ_prob": self.es_succ_prob, "es_degradation": self.es_degradation}
        rule = Rule(5, es_rule_actionA, es_rule_ASAP_conditionA, action_args, condition_args)
        rules.append(rule)
    elif index == 0:
        condition_args = {"memory_indices": memory_indices, "target_remote": path[-1], "fidelity": reservation.fidelity}
        action_args = {}
        rule = Rule(10, es_rule_actionB, es_rule_conditionB1, action_args, condition_args)
        rules.append(rule)

    elif index == len(path) - 1:
        action_args = {}
        condition_args = {"memory_indices": memory_indices, "target_remote": path[0], "fidelity": reservation.fidelity}
        rule = Rule(10, es_rule_actionB, es_rule_conditionB1, action_args, condition_args)
        rules.append(rule)

    else:
        # Modified logic based on the specified order
        node_index_in_order = reservation.swapping_order.index(path[index])

        # For the last node in the order list
        if node_index_in_order == len(reservation.swapping_order) - 1:
            left, right = path[0], path[-1]
        else:
            # Find the nearest node from the left that is not before the current node in the order list
            left = next(
                (node for node in reversed(path[:index]) if node not in reservation.swapping_order[:node_index_in_order]),
                None,
            )

            # Find the nearest node from the right that is not before the current node in the order list
            right = next(
                (node for node in path[index + 1 :] if node not in reservation.swapping_order[:node_index_in_order]), None
            )

        condition_args = {"memory_indices": memory_indices, "left": left, "right": right, "fidelity": reservation.fidelity}
        action_args = {"es_succ_prob": self.es_succ_prob, "es_degradation": self.es_degradation}
        rule = Rule(5, es_rule_actionA, es_rule_conditionA, action_args, condition_args)
        rules.append(rule)

        action_args = {}
        rule = Rule(10, es_rule_actionB, es_rule_conditionB2, action_args, condition_args)
        rules.append(rule)

    for rule in rules:
        rule.set_reservation(reservation)

    return rules
