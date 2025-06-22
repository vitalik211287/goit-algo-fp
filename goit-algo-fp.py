# # ===============================
# # 🌟 Task 1: Singly Linked List Operations
# # ===============================

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None

# class LinkedList:
#     def __init__(self):
#         self.head = None

#     def append(self, data):
#         new_node = Node(data)
#         if not self.head:
#             self.head = new_node
#             return
#         current = self.head
#         while current.next:
#             current = current.next
#         current.next = new_node

#     def reverse(self):
#         prev = None
#         current = self.head
#         while current:
#             next_node = current.next
#             current.next = prev
#             prev = current
#             current = next_node
#         self.head = prev

#     def to_list(self):
#         result = []
#         current = self.head
#         while current:
#             result.append(current.data)
#             current = current.next
#         return result

#     def sorted_merge(self, other):
#         dummy = Node(0)
#         tail = dummy
#         a = self.head
#         b = other.head
#         while a and b:
#             if a.data < b.data:
#                 tail.next = a
#                 a = a.next
#             else:
#                 tail.next = b
#                 b = b.next
#             tail = tail.next
#         tail.next = a or b
#         merged = LinkedList()
#         merged.head = dummy.next
#         return merged

#     def merge_sort(self):
#         if not self.head or not self.head.next:
#             return self

#         def split(head):
#             slow = head
#             fast = head.next
#             while fast and fast.next:
#                 slow = slow.next
#                 fast = fast.next.next
#             middle = slow.next
#             slow.next = None
#             return head, middle

#         def merge(left, right):
#             dummy = Node(0)
#             tail = dummy
#             while left and right:
#                 if left.data < right.data:
#                     tail.next = left
#                     left = left.next
#                 else:
#                     tail.next = right
#                     right = right.next
#                 tail = tail.next
#             tail.next = left or right
#             return dummy.next

#         def merge_sort_rec(head):
#             if not head or not head.next:
#                 return head
#             left_head, right_head = split(head)
#             left = merge_sort_rec(left_head)
#             right = merge_sort_rec(right_head)
#             return merge(left, right)

#         self.head = merge_sort_rec(self.head)
#         return self


# # ===============================
# # 🌳 Task 2: Pythagoras Tree (Fractal)
# # ===============================

# import turtle
# import math

# def draw_tree(length, level):
#     if level == 0:
#         return
#     turtle.forward(length)
#     turtle.left(45)
#     draw_tree(length * math.sqrt(2) / 2, level - 1)
#     turtle.right(90)
#     draw_tree(length * math.sqrt(2) / 2, level - 1)
#     turtle.left(45)
#     turtle.backward(length)


# def pythagoras_tree(level):
#     turtle.speed(0)
#     turtle.left(90)
#     draw_tree(100, level)
#     turtle.done()


# # ===============================
# # 🤷️‍♂️ Task 3: Dijkstra Algorithm using Heap
# # ===============================

# import heapq

# def dijkstra(graph, start):
#     distances = {vertex: float('inf') for vertex in graph}
#     distances[start] = 0
#     queue = [(0, start)]
#     while queue:
#         current_distance, current_vertex = heapq.heappop(queue)
#         if current_distance > distances[current_vertex]:
#             continue
#         for neighbor, weight in graph[current_vertex].items():
#             distance = current_distance + weight
#             if distance < distances[neighbor]:
#                 distances[neighbor] = distance
#                 heapq.heappush(queue, (distance, neighbor))
#     return distances


# # ===============================
# # 📏 Task 4: Visualize Binary Heap as Tree
# # ===============================

# import networkx as nx
# import matplotlib.pyplot as plt
# import uuid

# class TreeNode:
#     def __init__(self, val):
#         self.left = None
#         self.right = None
#         self.val = val
#         self.id = str(uuid.uuid4())


# def build_heap_tree(heap, i=0):
#     if i >= len(heap):
#         return None
#     node = TreeNode(heap[i])
#     node.left = build_heap_tree(heap, 2 * i + 1)
#     node.right = build_heap_tree(heap, 2 * i + 2)
#     return node


# def add_edges(graph, node, pos, x=0, y=0, layer=1):
#     if node:
#         graph.add_node(node.id, label=node.val)
#         pos[node.id] = (x, y)
#         if node.left:
#             graph.add_edge(node.id, node.left.id)
#             add_edges(graph, node.left, pos, x - 1 / 2 ** layer, y - 1, layer + 1)
#         if node.right:
#             graph.add_edge(node.id, node.right.id)
#             add_edges(graph, node.right, pos, x + 1 / 2 ** layer, y - 1, layer + 1)


# def draw_heap(heap):
#     root = build_heap_tree(heap)
#     graph = nx.DiGraph()
#     pos = {}
#     add_edges(graph, root, pos)
#     labels = nx.get_node_attributes(graph, 'label')
#     nx.draw(graph, pos, labels=labels, with_labels=True, node_size=2000, node_color="skyblue")
#     plt.show()


# # ===============================
# # 🔄 Task 5: Tree Traversal Visualization (DFS & BFS)
# # ===============================

# from collections import deque

# def visualize_traversal(root, mode='dfs'):
#     tree = nx.DiGraph()
#     pos = {}
#     add_edges(tree, root, pos)
#     nodes = list(tree.nodes)
#     visited = set()
#     color_map = {}

#     if mode == 'dfs':
#         stack = [root]
#         order = []
#         while stack:
#             node = stack.pop()
#             if node.id not in visited:
#                 visited.add(node.id)
#                 order.append(node.id)
#                 if node.right:
#                     stack.append(node.right)
#                 if node.left:
#                     stack.append(node.left)
#     elif mode == 'bfs':
#         queue = deque([root])
#         order = []
#         while queue:
#             node = queue.popleft()
#             if node.id not in visited:
#                 visited.add(node.id)
#                 order.append(node.id)
#                 if node.left:
#                     queue.append(node.left)
#                 if node.right:
#                     queue.append(node.right)
#     else:
#         return

#     for i, node_id in enumerate(order):
#         intensity = hex(16 + i * 10)[2:] if (16 + i * 10) < 256 else "ff"
#         color_map[node_id] = f"#{intensity}96F0"

#     colors = [color_map.get(n, "#1296F0") for n in nodes]
#     labels = nx.get_node_attributes(tree, 'label')
#     nx.draw(tree, pos, labels=labels, with_labels=True, node_size=2000, node_color=colors)
#     plt.title(f"{mode.upper()} traversal")
#     plt.show()


# # ===============================
# # 🍔 Task 6: Greedy & Dynamic Programming
# # ===============================

# def greedy_algorithm(items, budget):
#     sorted_items = sorted(items.items(), key=lambda x: x[1]['calories'] / x[1]['cost'], reverse=True)
#     total_cost = 0
#     result = []
#     for name, data in sorted_items:
#         if total_cost + data['cost'] <= budget:
#             result.append(name)
#             total_cost += data['cost']
#     return result


# def dynamic_programming(items, budget):
#     names = list(items.keys())
#     dp = [0] * (budget + 1)
#     keep = [None] * (budget + 1)
#     for i, name in enumerate(names):
#         cost = items[name]['cost']
#         calories = items[name]['calories']
#         for b in range(budget, cost - 1, -1):
#             if dp[b - cost] + calories > dp[b]:
#                 dp[b] = dp[b - cost] + calories
#                 keep[b] = i

#     result = []
#     b = budget
#     while b >= 0 and keep[b] is not None:
#         i = keep[b]
#         name = names[i]
#         result.append(name)
#         b -= items[name]['cost']
#     return result[::-1]


# # ===============================
# # 🤺 Task 7: Monte Carlo Dice Simulation
# # ===============================

# import random
# import pandas as pd
# import matplotlib.pyplot as plt 

# def monte_carlo_simulation(throws=10000):
#     counts = {i: 0 for i in range(2, 13)}
#     for _ in range(throws):
#         dice_sum = random.randint(1, 6) + random.randint(1, 6)
#         counts[dice_sum] += 1
#     probabilities = {k: v / throws for k, v in counts.items()}
#     df = pd.DataFrame(list(probabilities.items()), columns=['Sum', 'Probability'])
#     df.set_index('Sum', inplace=True)
#     df.plot(kind='bar', legend=False, title="Monte Carlo Dice Sum Probabilities")
#     plt.ylabel("Probability")
#     plt.show()
#     return df
# ===============================
# 🌟 Task 1: Singly Linked List Operations
# ===============================

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None

# class LinkedList:
#     def __init__(self):
#         self.head = None

#     def append(self, data):
#         new_node = Node(data)
#         if not self.head:
#             self.head = new_node
#             return
#         current = self.head
#         while current.next:
#             current = current.next
#         current.next = new_node

#     def reverse(self):
#         prev = None
#         current = self.head
#         while current:
#             next_node = current.next
#             current.next = prev
#             prev = current
#             current = next_node
#         self.head = prev

#     def to_list(self):
#         result = []
#         current = self.head
#         while current:
#             result.append(current.data)
#             current = current.next
#         return result

#     def sorted_merge(self, other):
#         dummy = Node(0)
#         tail = dummy
#         a = self.head
#         b = other.head
#         while a and b:
#             if a.data < b.data:
#                 tail.next = a
#                 a = a.next
#             else:
#                 tail.next = b
#                 b = b.next
#             tail = tail.next
#         tail.next = a or b
#         merged = LinkedList()
#         merged.head = dummy.next
#         return merged

#     def merge_sort(self):
#         if not self.head or not self.head.next:
#             return self

#         def split(head):
#             slow = head
#             fast = head.next
#             while fast and fast.next:
#                 slow = slow.next
#                 fast = fast.next.next
#             middle = slow.next
#             slow.next = None
#             return head, middle

#         def merge(left, right):
#             dummy = Node(0)
#             tail = dummy
#             while left and right:
#                 if left.data < right.data:
#                     tail.next = left
#                     left = left.next
#                 else:
#                     tail.next = right
#                     right = right.next
#                 tail = tail.next
#             tail.next = left or right
#             return dummy.next

#         def merge_sort_rec(head):
#             if not head or not head.next:
#                 return head
#             left_head, right_head = split(head)
#             left = merge_sort_rec(left_head)
#             right = merge_sort_rec(right_head)
#             return merge(left, right)

#         self.head = merge_sort_rec(self.head)
#         return self

# if __name__ == "__main__":
#     print("\n=== Task 1: Singly Linked List Operations ===")
#     ll = LinkedList()
#     for val in [3, 1, 4, 2]:
#         ll.append(val)
#     print("Original:", ll.to_list())
#     ll.reverse()
#     print("Reversed:", ll.to_list())
#     ll.merge_sort()
#     print("Sorted:", ll.to_list())

#     ll2 = LinkedList()
#     for val in [5, 6]:
#         ll2.append(val)
#     merged = ll.sorted_merge(ll2)
#     print("Merged:", merged.to_list())


# ===============================
# 🌟 Task 1: Singly Linked List Operations
# ===============================

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None

# class LinkedList:
#     def __init__(self):
#         self.head = None

#     def append(self, data):
#         new_node = Node(data)
#         if not self.head:
#             self.head = new_node
#             return
#         current = self.head
#         while current.next:
#             current = current.next
#         current.next = new_node

#     def to_list(self):
#         result = []
#         current = self.head
#         while current:
#             result.append(current.data)
#             current = current.next
#         return result

#     def reverse(self):
#         prev = None
#         current = self.head
#         while current:
#             next_node = current.next
#             current.next = prev
#             prev = current
#             current = next_node
#         self.head = prev

# if __name__ == "__main__":
#     print("\n=== Task 1: Singly Linked List Operations ===")
#     ll = LinkedList()
#     for i in [3, 1, 4, 1, 5]:
#         ll.append(i)
#     print("Original List:", ll.to_list())
#     ll.reverse()
#     print("Reversed List:", ll.to_list())


# # ===============================
# # 🌱 Task 2: Pythagoras Tree (Fractal)
# # ===============================

# import turtle
# import math

# def draw_tree(length, level):
#     if level == 0:
#         return
#     turtle.forward(length)
#     turtle.left(45)
#     draw_tree(length * math.sqrt(2) / 2, level - 1)
#     turtle.right(90)
#     draw_tree(length * math.sqrt(2) / 2, level - 1)
#     turtle.left(45)
#     turtle.backward(length)

# def pythagoras_tree(level):
#     turtle.speed(0)
#     turtle.left(90)
#     draw_tree(100, level)
#     turtle.done()


# # ===============================
# # 🧐 Task 3: Dijkstra Algorithm using Heap
# # ===============================

# import heapq

# def dijkstra(graph, start):
#     distances = {vertex: float('inf') for vertex in graph}
#     distances[start] = 0
#     queue = [(0, start)]
#     while queue:
#         current_distance, current_vertex = heapq.heappop(queue)
#         if current_distance > distances[current_vertex]:
#             continue
#         for neighbor, weight in graph[current_vertex].items():
#             distance = current_distance + weight
#             if distance < distances[neighbor]:
#                 distances[neighbor] = distance
#                 heapq.heappush(queue, (distance, neighbor))
#     return distances

# if __name__ == "__main__":
#     graph = {
#         'A': {'B': 1, 'C': 4},
#         'B': {'C': 2, 'D': 5},
#         'C': {'D': 1},
#         'D': {}
#     }
#     print("\n=== Task 3: Dijkstra Algorithm ===")
#     print("Distances:", dijkstra(graph, 'A'))


# # ===============================
# # 📊 Task 4: Visualize Binary Heap as Tree
# # ===============================

# import networkx as nx
# import matplotlib.pyplot as plt
# import uuid

# class TreeNode:
#     def __init__(self, val):
#         self.left = None
#         self.right = None
#         self.val = val
#         self.id = str(uuid.uuid4())

# def build_heap_tree(heap, i=0):
#     if i >= len(heap):
#         return None
#     node = TreeNode(heap[i])
#     node.left = build_heap_tree(heap, 2 * i + 1)
#     node.right = build_heap_tree(heap, 2 * i + 2)
#     return node

# def add_edges(graph, node, pos, x=0, y=0, layer=1):
#     if node:
#         graph.add_node(node.id, label=node.val)
#         pos[node.id] = (x, y)
#         if node.left:
#             graph.add_edge(node.id, node.left.id)
#             add_edges(graph, node.left, pos, x - 1 / 2 ** layer, y - 1, layer + 1)
#         if node.right:
#             graph.add_edge(node.id, node.right.id)
#             add_edges(graph, node.right, pos, x + 1 / 2 ** layer, y - 1, layer + 1)

# def draw_heap(heap):
#     root = build_heap_tree(heap)
#     graph = nx.DiGraph()
#     pos = {}
#     add_edges(graph, root, pos)
#     labels = nx.get_node_attributes(graph, 'label')
#     nx.draw(graph, pos, labels=labels, with_labels=True, node_size=2000, node_color="skyblue")
#     plt.show()


# # ===============================
# # 🔄 Task 5: Tree Traversal Visualization (DFS & BFS)
# # ===============================

# from collections import deque

# def visualize_traversal(root, mode='dfs'):
#     tree = nx.DiGraph()
#     pos = {}
#     add_edges(tree, root, pos)
#     nodes = list(tree.nodes)
#     visited = set()
#     color_map = {}

#     if mode == 'dfs':
#         stack = [root]
#         order = []
#         while stack:
#             node = stack.pop()
#             if node.id not in visited:
#                 visited.add(node.id)
#                 order.append(node.id)
#                 if node.right:
#                     stack.append(node.right)
#                 if node.left:
#                     stack.append(node.left)
#     elif mode == 'bfs':
#         queue = deque([root])
#         order = []
#         while queue:
#             node = queue.popleft()
#             if node.id not in visited:
#                 visited.add(node.id)
#                 order.append(node.id)
#                 if node.left:
#                     queue.append(node.left)
#                 if node.right:
#                     queue.append(node.right)

#     for i, node_id in enumerate(order):
#         intensity = hex(16 + i * 10)[2:] if (16 + i * 10) < 256 else "ff"
#         color_map[node_id] = f"#{intensity}96F0"

#     colors = [color_map.get(n, "#1296F0") for n in nodes]
#     labels = nx.get_node_attributes(tree, 'label')
#     nx.draw(tree, pos, labels=labels, with_labels=True, node_size=2000, node_color=colors)
#     plt.title(f"{mode.upper()} traversal")
#     plt.show()


# # ===============================
# # 🍔 Task 6: Greedy & Dynamic Programming
# # ===============================

# def greedy_algorithm(items, budget):
#     sorted_items = sorted(items.items(), key=lambda x: x[1]['calories'] / x[1]['cost'], reverse=True)
#     total_cost = 0
#     result = []
#     for name, data in sorted_items:
#         if total_cost + data['cost'] <= budget:
#             result.append(name)
#             total_cost += data['cost']
#     return result

# def dynamic_programming(items, budget):
#     names = list(items.keys())
#     dp = [0] * (budget + 1)
#     keep = [None] * (budget + 1)
#     for i, name in enumerate(names):
#         cost = items[name]['cost']
#         calories = items[name]['calories']
#         for b in range(budget, cost - 1, -1):
#             if dp[b - cost] + calories > dp[b]:
#                 dp[b] = dp[b - cost] + calories
#                 keep[b] = i

#     result = []
#     b = budget
#     while b >= 0 and keep[b] is not None:
#         i = keep[b]
#         name = names[i]
#         result.append(name)
#         b -= items[name]['cost']
#     return result[::-1]

# if __name__ == "__main__":
#     print("\n=== Task 6: Greedy & Dynamic Programming ===")
#     items = {
#         'apple': {'cost': 2, 'calories': 100},
#         'banana': {'cost': 1, 'calories': 80},
#         'burger': {'cost': 5, 'calories': 500},
#     }
#     budget = 5
#     print("Greedy:", greedy_algorithm(items, budget))
#     print("Dynamic Programming:", dynamic_programming(items, budget))


# # ===============================
# # ⚔️ Task 7: Monte Carlo Dice Simulation
# # ===============================

# import random
# import pandas as pd

# def monte_carlo_simulation(throws=10000):
#     counts = {i: 0 for i in range(2, 13)}
#     for _ in range(throws):
#         dice_sum = random.randint(1, 6) + random.randint(1, 6)
#         counts[dice_sum] += 1
#     probabilities = {k: v / throws for k, v in counts.items()}
#     df = pd.DataFrame(list(probabilities.items()), columns=['Sum', 'Probability'])
#     return df

# if __name__ == "__main__":
#     print("\n=== Task 7: Monte Carlo Dice Simulation ===")
#     print(monte_carlo_simulation(1000))


# ===============================
# 🌟 Task 1: Singly Linked List Operations
# ===============================

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None

# class LinkedList:
#     def __init__(self):
#         self.head = None

#     def append(self, data):
#         new_node = Node(data)
#         if not self.head:
#             self.head = new_node
#             return
#         current = self.head
#         while current.next:
#             current = current.next
#         current.next = new_node

#     def reverse(self):
#         prev = None
#         current = self.head
#         while current:
#             next_node = current.next
#             current.next = prev
#             prev = current
#             current = next_node
#         self.head = prev

#     def to_list(self):
#         result = []
#         current = self.head
#         while current:
#             result.append(current.data)
#             current = current.next
#         return result

#     def merge_sort(self):
#         if not self.head or not self.head.next:
#             return self

#         def split(head):
#             slow = head
#             fast = head.next
#             while fast and fast.next:
#                 slow = slow.next
#                 fast = fast.next.next
#             middle = slow.next
#             slow.next = None
#             return head, middle

#         def merge(left, right):
#             dummy = Node(0)
#             tail = dummy
#             while left and right:
#                 if left.data < right.data:
#                     tail.next = left
#                     left = left.next
#                 else:
#                     tail.next = right
#                     right = right.next
#                 tail = tail.next
#             tail.next = left or right
#             return dummy.next

#         def merge_sort_rec(head):
#             if not head or not head.next:
#                 return head
#             left_head, right_head = split(head)
#             left = merge_sort_rec(left_head)
#             right = merge_sort_rec(right_head)
#             return merge(left, right)

#         self.head = merge_sort_rec(self.head)
#         return self

# if __name__ == "__main__":
#     print("\n=== Task 1: Linked List Merge Sort ===")
#     ll = LinkedList()
#     for val in [4, 2, 5, 1, 3]:
#         ll.append(val)
#     print("Original list:", ll.to_list())
#     ll.merge_sort()
#     print("Sorted list:", ll.to_list())


# # ===============================
# # 🌱 Task 2: Pythagoras Tree (Fractal)
# # ===============================

# import turtle
# import math

# def draw_tree(length, level):
#     if level == 0:
#         return
#     turtle.forward(length)
#     turtle.left(45)
#     draw_tree(length * math.sqrt(2) / 2, level - 1)
#     turtle.right(90)
#     draw_tree(length * math.sqrt(2) / 2, level - 1)
#     turtle.left(45)
#     turtle.backward(length)

# def pythagoras_tree(level):
#     turtle.speed(0)
#     turtle.left(90)
#     draw_tree(100, level)
#     turtle.done()
# if __name__ == "__main__":
#     print("\n=== Task 2: Pythagoras Tree (Fractal) ===")
#     try:
#         pythagoras_tree(5)
#     except turtle.Terminator:
#         print("Pythagoras Tree window closed.")    


# # ===============================
# # 🤷️‍♂️ Task 3: Dijkstra Algorithm using Heap
# # ===============================

# import heapq

# def dijkstra(graph, start):
#     distances = {vertex: float('inf') for vertex in graph}
#     distances[start] = 0
#     queue = [(0, start)]
#     while queue:
#         current_distance, current_vertex = heapq.heappop(queue)
#         if current_distance > distances[current_vertex]:
#             continue
#         for neighbor, weight in graph[current_vertex].items():
#             distance = current_distance + weight
#             if distance < distances[neighbor]:
#                 distances[neighbor] = distance
#                 heapq.heappush(queue, (distance, neighbor))
#     return distances

# if __name__ == "__main__":
#     print("\n=== Task 3: Dijkstra Algorithm ===")
#     graph = {
#         'A': {'B': 1, 'C': 4},
#         'B': {'C': 2, 'D': 5},
#         'C': {'D': 1},
#         'D': {}
#     }
#     result = dijkstra(graph, 'A')
#     print("Shortest paths from A:", result)


# # ===============================
# # 📏 Task 4: Visualize Binary Heap as Tree
# # ===============================

# import networkx as nx
# import matplotlib.pyplot as plt
# import uuid

# class TreeNode:
#     def __init__(self, val):
#         self.left = None
#         self.right = None
#         self.val = val
#         self.id = str(uuid.uuid4())

# def build_heap_tree(heap, i=0):
#     if i >= len(heap):
#         return None
#     node = TreeNode(heap[i])
#     node.left = build_heap_tree(heap, 2 * i + 1)
#     node.right = build_heap_tree(heap, 2 * i + 2)
#     return node

# def add_edges(graph, node, pos, x=0, y=0, layer=1):
#     if node:
#         graph.add_node(node.id, label=node.val)
#         pos[node.id] = (x, y)
#         if node.left:
#             graph.add_edge(node.id, node.left.id)
#             add_edges(graph, node.left, pos, x - 1 / 2 ** layer, y - 1, layer + 1)
#         if node.right:
#             graph.add_edge(node.id, node.right.id)
#             add_edges(graph, node.right, pos, x + 1 / 2 ** layer, y - 1, layer + 1)

# def draw_heap(heap):
#     root = build_heap_tree(heap)
#     graph = nx.DiGraph()
#     pos = {}
#     add_edges(graph, root, pos)
#     labels = nx.get_node_attributes(graph, 'label')
#     nx.draw(graph, pos, labels=labels, with_labels=True, node_size=2000, node_color="skyblue")
#     plt.title("Heap Visualization")
#     plt.show()


# # ===============================
# # 🔄 Task 5: Tree Traversal Visualization (DFS & BFS)
# # ===============================

# from collections import deque

# def visualize_traversal(root, mode='dfs'):
#     tree = nx.DiGraph()
#     pos = {}
#     add_edges(tree, root, pos)
#     nodes = list(tree.nodes)
#     visited = set()
#     color_map = {}

#     if mode == 'dfs':
#         stack = [root]
#         order = []
#         while stack:
#             node = stack.pop()
#             if node.id not in visited:
#                 visited.add(node.id)
#                 order.append(node.id)
#                 if node.right:
#                     stack.append(node.right)
#                 if node.left:
#                     stack.append(node.left)
#     elif mode == 'bfs':
#         queue = deque([root])
#         order = []
#         while queue:
#             node = queue.popleft()
#             if node.id not in visited:
#                 visited.add(node.id)
#                 order.append(node.id)
#                 if node.left:
#                     queue.append(node.left)
#                 if node.right:
#                     queue.append(node.right)
#     else:
#         return

#     for i, node_id in enumerate(order):
#         intensity = hex(16 + i * 10)[2:] if (16 + i * 10) < 256 else "ff"
#         color_map[node_id] = f"#{intensity}96F0"

#     colors = [color_map.get(n, "#1296F0") for n in nodes]
#     labels = nx.get_node_attributes(tree, 'label')
#     nx.draw(tree, pos, labels=labels, with_labels=True, node_size=2000, node_color=colors)
#     plt.title(f"{mode.upper()} traversal")
#     plt.show()


# # ===============================
# # 🍔 Task 6: Greedy & Dynamic Programming
# # ===============================

# def greedy_algorithm(items, budget):
#     sorted_items = sorted(items.items(), key=lambda x: x[1]['calories'] / x[1]['cost'], reverse=True)
#     total_cost = 0
#     result = []
#     for name, data in sorted_items:
#         if total_cost + data['cost'] <= budget:
#             result.append(name)
#             total_cost += data['cost']
#     return result

# def dynamic_programming(items, budget):
#     names = list(items.keys())
#     dp = [0] * (budget + 1)
#     keep = [None] * (budget + 1)
#     for i, name in enumerate(names):
#         cost = items[name]['cost']
#         calories = items[name]['calories']
#         for b in range(budget, cost - 1, -1):
#             if dp[b - cost] + calories > dp[b]:
#                 dp[b] = dp[b - cost] + calories
#                 keep[b] = i

#     result = []
#     b = budget
#     while b >= 0 and keep[b] is not None:
#         i = keep[b]
#         name = names[i]
#         result.append(name)
#         b -= items[name]['cost']
#     return result[::-1]

# if __name__ == "__main__":
#     print("\n=== Task 6: Knapsack Example ===")
#     food_items = {
#         'Burger': {'cost': 3, 'calories': 500},
#         'Salad': {'cost': 2, 'calories': 150},
#         'Fries': {'cost': 1, 'calories': 300},
#         'Steak': {'cost': 5, 'calories': 700}
#     }
#     budget = 6
#     print("Greedy result:", greedy_algorithm(food_items, budget))
#     print("DP result:", dynamic_programming(food_items, budget))


# # ===============================
# # 🎲 Task 7: Monte Carlo Dice Simulation
# # ===============================

# import random
# import pandas as pd

# def monte_carlo_simulation(throws=10000):
#     counts = {i: 0 for i in range(2, 13)}
#     for _ in range(throws):
#         dice_sum = random.randint(1, 6) + random.randint(1, 6)
#         counts[dice_sum] += 1
#     probabilities = {k: v / throws for k, v in counts.items()}
#     return probabilities

# if __name__ == "__main__":
#     print("\n=== Task 7: Monte Carlo Dice ===")
#     probs = monte_carlo_simulation(5000)
#     for k in sorted(probs):
#         print(f"Sum {k}: {probs[k]:.4f}")

# ===============================
# 🌄 Task 1: Singly Linked List Operations
# ===============================

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def reverse(self):
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def to_list(self):
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result

    def merge_sort(self):
        if not self.head or not self.head.next:
            return self

        def split(head):
            slow = head
            fast = head.next
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            middle = slow.next
            slow.next = None
            return head, middle

        def merge(left, right):
            dummy = Node(0)
            tail = dummy
            while left and right:
                if left.data < right.data:
                    tail.next = left
                    left = left.next
                else:
                    tail.next = right
                    right = right.next
                tail = tail.next
            tail.next = left or right
            return dummy.next

        def merge_sort_rec(head):
            if not head or not head.next:
                return head
            left_head, right_head = split(head)
            left = merge_sort_rec(left_head)
            right = merge_sort_rec(right_head)
            return merge(left, right)

        self.head = merge_sort_rec(self.head)
        return self

print("\n=== Task 1: Linked List Merge Sort ===")
ll = LinkedList()
for val in [4, 2, 5, 1, 3]:
    ll.append(val)
print("Original list:", ll.to_list())
ll.merge_sort()
print("Sorted list:", ll.to_list())


# ===============================
# 🌱 Task 2: Pythagoras Tree (Fractal)
# ===============================

import turtle
import math

def draw_tree(length, level):
    if level == 0:
        return
    turtle.forward(length)
    turtle.left(45)
    draw_tree(length * math.sqrt(2) / 2, level - 1)
    turtle.right(90)
    draw_tree(length * math.sqrt(2) / 2, level - 1)
    turtle.left(45)
    turtle.backward(length)

def pythagoras_tree(level):
    turtle.speed(0)
    turtle.left(90)
    draw_tree(100, level)
    turtle.done()

print("\n=== Task 2: Pythagoras Tree (Fractal) ===")
try:
    pythagoras_tree(5)
except turtle.Terminator:
    print("Pythagoras Tree window closed.")


# ===============================
# 🤷️‍♂️ Task 3: Dijkstra Algorithm using Heap
# ===============================

import heapq

def dijkstra(graph, start):
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    queue = [(0, start)]
    while queue:
        current_distance, current_vertex = heapq.heappop(queue)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances

print("\n=== Task 3: Dijkstra Algorithm ===")
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 2, 'D': 5},
    'C': {'D': 1},
    'D': {}
}
result = dijkstra(graph, 'A')
print("Shortest paths from A:", result)


# ===============================
# 📏 Task 4: Visualize Binary Heap as Tree
# ===============================

import networkx as nx
import matplotlib.pyplot as plt
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

def draw_heap(heap):
    root = build_heap_tree(heap)
    graph = nx.DiGraph()
    pos = {}
    add_edges(graph, root, pos)
    labels = nx.get_node_attributes(graph, 'label')
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=2000, node_color="skyblue")
    plt.title("Heap Visualization")
    plt.show()

print("\n=== Task 4: Draw Heap ===")
draw_heap([10, 5, 3, 2, 4, 1])


# ===============================
# 🔄 Task 5: Tree Traversal Visualization (DFS & BFS)
# ===============================

from collections import deque

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

print("\n=== Task 5: Traversal DFS ===")
root_node = build_heap_tree([10, 5, 3, 2, 4, 1])
visualize_traversal(root_node, mode='dfs')
print("\n=== Task 5: Traversal BFS ===")
visualize_traversal(root_node, mode='bfs')


# ===============================
# 🍔 Task 6: Greedy & Dynamic Programming
# ===============================

def greedy_algorithm(items, budget):
    sorted_items = sorted(items.items(), key=lambda x: x[1]['calories'] / x[1]['cost'], reverse=True)
    total_cost = 0
    result = []
    for name, data in sorted_items:
        if total_cost + data['cost'] <= budget:
            result.append(name)
            total_cost += data['cost']
    return result

def dynamic_programming(items, budget):
    names = list(items.keys())
    dp = [0] * (budget + 1)
    keep = [None] * (budget + 1)
    for i, name in enumerate(names):
        cost = items[name]['cost']
        calories = items[name]['calories']
        for b in range(budget, cost - 1, -1):
            if dp[b - cost] + calories > dp[b]:
                dp[b] = dp[b - cost] + calories
                keep[b] = i

    result = []
    b = budget
    while b >= 0 and keep[b] is not None:
        i = keep[b]
        name = names[i]
        result.append(name)
        b -= items[name]['cost']
    return result[::-1]

print("\n=== Task 6: Knapsack Example ===")
food_items = {
    'Burger': {'cost': 3, 'calories': 500},
    'Salad': {'cost': 2, 'calories': 150},
    'Fries': {'cost': 1, 'calories': 300},
    'Steak': {'cost': 5, 'calories': 700}
}
budget = 6
print("Greedy result:", greedy_algorithm(food_items, budget))
print("DP result:", dynamic_programming(food_items, budget))


# ===============================
# 🏋️‍♂️ Task 7: Monte Carlo Dice Simulation
# ===============================

import random
import pandas as pd

def monte_carlo_simulation(throws=10000):
    counts = {i: 0 for i in range(2, 13)}
    for _ in range(throws):
        dice_sum = random.randint(1, 6) + random.randint(1, 6)
        counts[dice_sum] += 1
    probabilities = {k: v / throws for k, v in counts.items()}
    return probabilities

print("\n=== Task 7: Monte Carlo Dice ===")
probs = monte_carlo_simulation(5000)
for k in sorted(probs):
    print(f"Sum {k}: {probs[k]:.4f}")

