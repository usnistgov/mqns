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

    def calculate_swapping_destinations(self, ranks):
        """Calculate Swapping Destinations (SD)."""
        SD_upstream = next((i for i in range(self.position - 1, -1, -1) if ranks[i] > self.rank), None)
        SD_downstream = next((i for i in range(self.position + 1, self.total_nodes) if ranks[i] > self.rank), None)
        return SD_upstream, SD_downstream

    def calculate_swapping_neighbors(self, ranks):
        """Calculate Swapping Neighbors (SN)."""
        SN_upstream = next((i for i in range(self.position - 1, -1, -1) if ranks[i] >= self.rank), None)
        SN_downstream = next((i for i in range(self.position + 1, self.total_nodes) if ranks[i] >= self.rank), None)
        return SN_upstream, SN_downstream

    def perform_operation(self):
        """Simulate the local operation."""
        self.result = f"Result@Node{self.position}"
        return self.result

    def process_update(self, message):
        """Process an incoming UPDATE message."""
        self.received_results.extend(message["results"])

    def send_update(self, direction, message):
        """Send an UPDATE message in a specified direction."""
        msg = {
            "from": self.position,
            "to": self.upstream if direction == "upstream" else self.downstream,
            "direction": direction,
            "results": message["results"].copy(),
        }
        print(f"MSG: {msg}")
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


def simulate_chain(rank_vector):
    total_nodes = len(rank_vector)
    nodes = [Node(i, rank_vector[i], total_nodes, rank_vector) for i in range(total_nodes)]
    messages = []

    # Determine how many results each node with rank > 0 needs to wait for
    waiting_count = [0] * total_nodes
    for i, rank in enumerate(rank_vector):
        if rank > 0:
            waiting_count[i] = sum(1 for j in range(total_nodes) if rank_vector[j] < rank)

    for node in nodes:
        if node.rank == 0:
            # print(f"node {node.position} (r={node.rank})")
            # Rank-0 node: Perform operation and send UPDATE to SN and SD
            node.perform_operation()
            SD_upstream, SD_downstream = node.calculate_swapping_destinations(rank_vector)
            SN_upstream, SN_downstream = node.calculate_swapping_neighbors(rank_vector)

            if ( (SN_upstream == SD_upstream and SN_upstream is not None)
                or (SN_downstream == SD_downstream and SN_downstream is not None) ):
                print(f"Node {node.position} (r={node.rank}) swap and generate message")
                dests = node.find_closest_rank()
                # print(dests)
                message = {"results": [node.result]}
                for d in dests:
                    if d < node.position:
                        messages.append(node.send_update("upstream", message))
                    else:
                        messages.append(node.send_update("downstream", message))               
            else:
                print(f"Node {node.position} (r={node.rank}) swap but does not generate message")


    # Process messages in transit
    while messages:
        new_messages = []
        for message in messages:
            to_node = message["to"]
            receiving_node = nodes[to_node]

            if receiving_node.position == 0 or receiving_node.position == total_nodes - 1:
                print(f"Reached end-node {receiving_node.position}")
                continue
                
            # if node already did swapping: forward or generate message
            if receiving_node.waiting_messages == [] and receiving_node.result is not None:
                print(f"Node {receiving_node.position}: merge and forward passing message")
                direction = message["direction"]
                message["results"].append(receiving_node.result)
                new_message = receiving_node.send_update(direction, message)
                new_messages.append(new_message)
                continue

            # Node has not done swapping yet:
            #   If node does not have conditions to swap -> store messages
            if len(receiving_node.received_results) < waiting_count[to_node]:
                print(f"Node {receiving_node.position} still waiting for results. Received one more update.")
                receiving_node.waiting_messages.append(message)
                receiving_node.process_update(message)
                
            #   If node has conditions to swap
            if len(receiving_node.received_results) >= waiting_count[to_node] and receiving_node.result is None:
                print(f"  Node {receiving_node.position} (r={receiving_node.rank}) swap")
                    
                receiving_node.perform_operation()
                    
                # process waiting messages
                for msg in receiving_node.waiting_messages:
                    print(f"Node {receiving_node.position}: merge and forward waiting message")
                    direction = msg["direction"]
                    msg["results"].append(receiving_node.result)

                    # for now, drop this message if this node is next to end-node and results not complete
                    if receiving_node.position == 1 or receiving_node.position == total_nodes - 2:
                        if len(msg["results"]) == len(rank_vector) - 2:
                            new_message = receiving_node.send_update(direction, msg)
                            new_messages.append(new_message)
                        else:
                            print(f"  Node {receiving_node.position}: drop this message")
                    else:
                        new_message = receiving_node.send_update(direction, msg)
                        new_messages.append(new_message)

                receiving_node.waiting_messages = []
                    
                # if next to end-node -> generate msg in opposite direction + insert self result only
                if receiving_node.position == 1 or receiving_node.position == total_nodes - 2:
                    print(f"Node {receiving_node.position}: generate new message")
                    direction = "downstream" if receiving_node.position == 1 else "upstream"
                    msg = { "results": [receiving_node.result] }
                    new_message = receiving_node.send_update(direction, msg)
                    new_messages.append(new_message)

        messages = new_messages


# Example: Chain of 5 nodes with a rank vector
#rank_vector = [2, 0, 1, 0, 2]       # doubling
#rank_vector = [3, 0, 1, 2, 3]       # sequential 
#rank_vector = [1, 0, 0, 0, 1]       # parallel
#rank_vector = [2, 1, 0, 1, 2]       # other
rank_vector = [3, 0, 2, 0, 1, 0, 3]
simulate_chain(rank_vector)
