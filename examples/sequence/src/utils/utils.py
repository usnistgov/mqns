import hashlib
import json


def list_to_swapping_order(swap_tree):
    result = []
    for item in swap_tree:
        if isinstance(item, int):
            result.append("r" + str(item))
        else:
            for element in item:
                result.append("r" + str(element))
    return result


def compute_swapping_strategy(number_of_routers, strategy_type):
    if strategy_type == "doubling":

        def balanced_tree_order(routers):
            order = []
            remaining_routers = []
            if not routers:
                return []
            for i in range(len(routers)):
                if i % 2 == 0:
                    order.append(routers[i])
                else:
                    remaining_routers.append(routers[i])
            return order + balanced_tree_order(remaining_routers)

        routers = [f"r{i + 1}" for i in range(number_of_routers)]
        return balanced_tree_order(routers)

    elif strategy_type == "sequential":
        return [f"r{i + 1}" for i in range(number_of_routers)]
    elif strategy_type in ["ASAP"]:
        return strategy_type
    else:
        raise ValueError(f"Invalid swapping strategy type: {strategy_type}")


def generate_sim_id(params):
    # Serialize parameters to a string and hash it
    params_string = json.dumps(params, sort_keys=True)  # Ensure consistent order
    hash_object = hashlib.md5(params_string.encode())
    sim_id = hash_object.hexdigest()[:8]
    return sim_id
