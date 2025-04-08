import copy

# Global log for visualization
visualization_events = []

class Node:
    def __init__(self, position, rank, total_nodes, rank_vector):
        self.rank_vector = rank_vector
        self.position = position
        self.rank = rank
        self.total_nodes = total_nodes
        self.upstream = position - 1 if position > 0 else None
        self.downstream = position + 1 if position < total_nodes - 1 else None
        self.result = None
        self.received_results = []
        self.waiting_messages = []
        self.just_performed_operation = False  # Track if operation just performed for message coloring

    def calculate_swapping_destinations(self, ranks):
        SD_upstream = next((i for i in range(self.position - 1, -1, -1) if ranks[i] > self.rank), None)
        SD_downstream = next((i for i in range(self.position + 1, self.total_nodes) if ranks[i] > self.rank), None)
        return SD_upstream, SD_downstream

    def calculate_swapping_neighbors(self, ranks):
        SN_upstream = next((i for i in range(self.position - 1, -1, -1) if ranks[i] >= self.rank), None)
        SN_downstream = next((i for i in range(self.position + 1, self.total_nodes) if ranks[i] >= self.rank), None)
        return SN_upstream, SN_downstream

    def perform_operation(self):
        # Simulate the local operation (SWAP)
        self.result = f"Result@Node{self.position}"
        self.just_performed_operation = True
        # Log the operation event
        visualization_events.append({
            "type": "operation",
            "node": self.position
        })
        return self.result

    def process_update(self, message):
        self.received_results.extend(message["results"])

    def send_update(self, direction, message):
        msg = {
            "from": self.position,
            "to": self.upstream if direction == "upstream" else self.downstream,
            "direction": direction,
            "results": message["results"].copy(),
        }

        # Determine if message is generated or forwarded
        # - If just_performed_operation is True at send time and the message has only this node's result,
        #   consider it a generated message (red arrow).
        # - Otherwise, consider it a forwarded message (black arrow).
        is_generated = False
        if self.just_performed_operation and len(msg["results"]) == 1 and f"Result@Node{self.position}" in msg["results"]:
            is_generated = True

        visualization_events.append({
            "type": "message",
            "from": msg["from"],
            "to": msg["to"],
            "direction": direction,
            "results": msg["results"].copy(),
            "generated": is_generated
        })

        # After sending a message, reset the flag
        self.just_performed_operation = False

        return msg
    
    def find_closest_rank(self):
        total_nodes = len(self.rank_vector)
        closest_positions = []
        closest_distance = float('inf')

        # Check immediate upstream neighbor
        if self.position > 0:
            upstream_position = self.position - 1
            upstream_distance = abs(self.rank_vector[upstream_position] - self.rank)
            if upstream_distance < closest_distance:
                closest_positions = [upstream_position]
                closest_distance = upstream_distance
            elif upstream_distance == closest_distance:
                closest_positions.append(upstream_position)

        # Check immediate downstream neighbor
        if self.position < total_nodes - 1:
            downstream_position = self.position + 1
            downstream_distance = abs(self.rank_vector[downstream_position] - self.rank)
            if downstream_distance < closest_distance:
                closest_positions = [downstream_position]
                closest_distance = downstream_distance
            elif downstream_distance == closest_distance:
                closest_positions.append(downstream_position)

        return closest_positions


def print_sequence_diagram(rank_vector, events):
    """
    Print a textual sequence diagram of the operations and message flow.
    We'll show nodes horizontally and events step by step.
    - Diamond (◊) for operation at a node.
    - (RED) arrow for generated messages
    - (BLACK) arrow for forwarded messages
    """
    nodes = len(rank_vector)
    # Headers
    header = "  ".join([f"N{n}(r={rank_vector[n]})" for n in range(nodes)])
    print("\nSequence Diagram:\n")
    print(header)
    print("-" * (len(header) + (nodes-1)*2))

    # We'll print events in order. Each event is either operation or message.
    # For each event, we print a row showing what's happening.
    # We'll represent time steps as separate lines.
    for i, ev in enumerate(events, start=1):
        line = ["   "] * nodes

        if ev["type"] == "operation":
            # Place a diamond at the node position
            node_pos = ev["node"]
            line[node_pos] = " ◊ "
            print("  ".join(line) + f"   (Operation at N{node_pos})")

        elif ev["type"] == "message":
            f = ev["from"]
            t = ev["to"]
            arrow_color = "(RED)" if ev["generated"] else "(BLACK)"
            arrow = "-->" if t > f else "<--"
            content = f"[MSG{i}:{','.join(ev['results'])}]"
            # Mark from and to:
            # We'll place markers [f] and [t] at respective columns
            for col in range(nodes):
                if col == f:
                    line[col] = f"[{f}]"
                elif col == t:
                    line[col] = f"[{t}]"
            # Print the message line
            print("  ".join(line) + f"  {arrow_color}{arrow}{content}")


def simulate_chain(rank_vector):
    total_nodes = len(rank_vector)
    nodes = [Node(i, rank_vector[i], total_nodes, rank_vector) for i in range(total_nodes)]
    messages = []

    waiting_count = [0] * total_nodes
    for i, rank in enumerate(rank_vector):
        if rank > 0:
            waiting_count[i] = sum(1 for j in range(total_nodes) if rank_vector[j] < rank)

    for node in nodes:
        if node.rank == 0:
            node.perform_operation()
            SD_upstream, SD_downstream = node.calculate_swapping_destinations(rank_vector)
            SN_upstream, SN_downstream = node.calculate_swapping_neighbors(rank_vector)

            if ((SN_upstream == SD_upstream and SN_upstream is not None)
                or (SN_downstream == SD_downstream and SN_downstream is not None)):
                dests = node.find_closest_rank()
                message = {"results": [node.result]}
                for d in dests:
                    if d < node.position:
                        messages.append(node.send_update("upstream", message))
                    else:
                        messages.append(node.send_update("downstream", message))               
            else:
                # Node did operation but does not generate message
                pass

    while messages:
        new_messages = []
        for message in messages:
            to_node = message["to"]
            receiving_node = nodes[to_node]

            if receiving_node.position == 0 or receiving_node.position == total_nodes - 1:
                # End node, message stops
                continue

            if receiving_node.waiting_messages == [] and receiving_node.result is not None:
                # already swapped, just merge and forward
                message["results"].append(receiving_node.result)
                new_message = receiving_node.send_update(message["direction"], message)
                new_messages.append(new_message)
                continue

            if len(receiving_node.received_results) < waiting_count[to_node]:
                receiving_node.waiting_messages.append(message)
                receiving_node.process_update(message)
                
            if len(receiving_node.received_results) >= waiting_count[to_node] and receiving_node.result is None:
                receiving_node.perform_operation()
                for msg in receiving_node.waiting_messages:
                    msg["results"].append(receiving_node.result)
                    if receiving_node.position == 1 or receiving_node.position == total_nodes - 2:
                        if len(msg["results"]) == len(rank_vector) - 2:
                            new_message = receiving_node.send_update(msg["direction"], msg)
                            new_messages.append(new_message)
                        else:
                            # drop this message
                            pass
                    else:
                        new_message = receiving_node.send_update(msg["direction"], msg)
                        new_messages.append(new_message)
                receiving_node.waiting_messages = []
                    
                if receiving_node.position == 1 or receiving_node.position == total_nodes - 2:
                    msg = { "results": [receiving_node.result] }
                    new_message = receiving_node.send_update("downstream" if receiving_node.position == 1 else "upstream", msg)
                    new_messages.append(new_message)

        messages = new_messages

# Example usage
rank_vector = [3, 0, 2, 0, 1, 0, 3]
simulate_chain(rank_vector)
print_sequence_diagram(rank_vector, visualization_events)
