from enum import Enum, auto
import random
import networkx as nx
import matplotlib.pyplot as plt
import time

class QubitState(Enum):
    ENTANGLED = auto()
    PURIF = auto()
    PENDING = auto()
    ELIGIBLE = auto()
    RELEASE = auto()

class QubitFSM:
    def __init__(self):
        self.state = QubitState.ENTANGLED
        self.purif_attempts = 0
        self.purif_success = False
        self.swapped = False

    def purify(self):
        if self.state == QubitState.ENTANGLED:    # swapping conditions met -> go to first purif (if any)
            self.state = QubitState.PURIF
            self.purif_attempts = 0
        elif self.state == QubitState.PENDING:    # pending purif succ -> go to next purif (if any)
            self.state = QubitState.PURIF
            self.purif_attempts +=1
    
    def start_purification(self):
        if self.state == QubitState.PURIF:
            if self.purif_success:
                self.state = QubitState.ELIGIBLE
            else:
                self.state = QubitState.ENTANGLED
        else:
            print(f"Unexpected transition: <{self.state}> -> <PENDING>")

    def swap(self):
        if self.state == QubitState.ELIGIBLE:
            if random.random() > 0.2:  # 80% success rate
                self.swapped = True
                self.state = QubitState.RELEASE
            else:
                self.state = QubitState.ENTANGLED

    def reset(self):
        self.__init__()

# Visualization function
def visualize_fsm(fsm, G, pos):
    plt.clf()
    node_colors = ['red' if state == fsm.state else 'lightblue' for state in QubitState]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, font_size=10, edge_color='gray')
    plt.pause(1)

# Create FSM graph
G = nx.DiGraph()
G.add_edges_from([
    (QubitState.ENTANGLED, QubitState.PURIF),
    (QubitState.PURIF, QubitState.PENDING),
    (QubitState.PENDING, QubitState.ELIGIBLE),
    (QubitState.ELIGIBLE, QubitState.RELEASE),
    (QubitState.PURIF, QubitState.ENTANGLED),
    (QubitState.ELIGIBLE, QubitState.ENTANGLED)
])
pos = nx.spring_layout(G)

# Run visualization
if __name__ == "__main__":
    fsm = QubitFSM()
    plt.figure(figsize=(6, 6))
    
    for _ in range(5):  # Run FSM steps with visualization
        visualize_fsm(fsm, G, pos)
        fsm.purify()
        visualize_fsm(fsm, G, pos)
        fsm.process_pending()
        visualize_fsm(fsm, G, pos)
        fsm.swap()
        visualize_fsm(fsm, G, pos)
        
        if fsm.state == QubitState.RELEASE:
            print("Qubit successfully delivered to user.")
            break
    
    plt.show()
