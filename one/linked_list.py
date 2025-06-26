class Node:
    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next

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

    def from_list(self, values):
        for value in values:
            self.append(value)

    def print(self):
        current = self.head
        while current:
            print(current.data, end=" → ")
            current = current.next
        print("None")

print("✅ linked_list.py імпортовано")