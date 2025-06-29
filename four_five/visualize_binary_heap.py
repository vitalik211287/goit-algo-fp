import networkx as nx
import matplotlib.pyplot as plt
from common import add_edges, build_heap_tree

def draw_heap(heap):
    root = build_heap_tree(heap)
    graph = nx.DiGraph()
    pos = {}
    add_edges(graph, root, pos)
    labels = nx.get_node_attributes(graph, 'label')
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=2000, node_color="skyblue")
    plt.title("Heap Visualization")
    plt.show()

print("\n=== Draw Heap ===")
draw_heap([10, 5, 3, 2, 4, 1])

