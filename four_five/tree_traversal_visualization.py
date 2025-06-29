from collections import deque
import networkx as nx
from matplotlib import pyplot as plt
from common import add_edges, build_heap_tree

def visualize_traversal(root, mode='dfs'):
    tree = nx.DiGraph()
    pos = {}
    add_edges(tree, root, pos)
    nodes = list(tree.nodes)
    visited = set()
    color_map = {}

    if mode == 'dfs':
        stack = [root]
        order = []
        while stack:
            node = stack.pop()
            if node.id not in visited:
                visited.add(node.id)
                order.append(node.id)
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
    elif mode == 'bfs':
        queue = deque([root])
        order = []
        while queue:
            node = queue.popleft()
            if node.id not in visited:
                visited.add(node.id)
                order.append(node.id)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
    else:
        return

    for i, node_id in enumerate(order):
        intensity = hex(16 + i * 10)[2:] if (16 + i * 10) < 256 else "ff"
        color_map[node_id] = f"#{intensity}96F0"

    colors = [color_map.get(n, "#1296F0") for n in nodes]
    labels = nx.get_node_attributes(tree, 'label')
    nx.draw(tree, pos, labels=labels, with_labels=True, node_size=2000, node_color=colors)
    plt.title(f"{mode.upper()} traversal")
    plt.show()

print("\n=== Traversal DFS ===")
root_node = build_heap_tree([10, 5, 3, 2, 4, 1])
visualize_traversal(root_node, mode='dfs')
print("\n=== Traversal BFS ===")
visualize_traversal(root_node, mode='bfs')