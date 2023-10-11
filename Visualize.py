import networkx as nx
import matplotlib.pyplot as plt
from Dense import Dense
from Activations import Tanh
import networkx as nx
import matplotlib.pyplot as plt


def visualize_network(network):
    G = nx.DiGraph()
    node_count = 0

    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            # Add nodes for the neurons in the current layer
            for _ in range(layer.num_neurons):
                G.add_node(node_count)
                node_count += 1

            # Add edges from the neurons in the previous layer to the neurons in the current layer
            if i > 0:
                for src in range(node_count - layer.num_neurons, node_count):
                    for dst in range(node_count, node_count + layer.input_dim):
                        G.add_edge(src, dst)

    nx.draw(G, with_labels=True)
    plt.show()

network = [
    Dense(5,10),
    Tanh(),
    Dense(10, 5),
    Tanh()
]

# Example usage
visualize_network(network)
