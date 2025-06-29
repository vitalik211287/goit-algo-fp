import uuid

class TreeNode:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val
        self.id = str(uuid.uuid4())

def build_heap_tree(heap, i=0):
    if i >= len(heap):
        return None
    node = TreeNode(heap[i])
    node.left = build_heap_tree(heap, 2 * i + 1)
    node.right = build_heap_tree(heap, 2 * i + 2)
    return node

def add_edges(graph, node, pos, x=0, y=0, layer=1):
    if node:
        graph.add_node(node.id, label=node.val)
        pos[node.id] = (x, y)
        if node.left:
            graph.add_edge(node.id, node.left.id)
            add_edges(graph, node.left, pos, x - 1 / 2 ** layer, y - 1, layer + 1)
        if node.right:
            graph.add_edge(node.id, node.right.id)
            add_edges(graph, node.right, pos, x + 1 / 2 ** layer, y - 1, layer + 1)