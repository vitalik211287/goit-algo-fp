from linked_list import LinkedList


def insertion_sort_linked_list(head):
    if not head or not head.next:
        return head

    sorted_head = None
    current = head

    while current:
        next_node = current.next
        sorted_head = insert_in_sorted_order(sorted_head, current)
        current = next_node

    return sorted_head

def insert_in_sorted_order(sorted_head, node_to_insert):
    if not sorted_head or node_to_insert.data < sorted_head.data:
        node_to_insert.next = sorted_head
        return node_to_insert

    current = sorted_head
    while current.next and current.next.data < node_to_insert.data:
        current = current.next

    node_to_insert.next = current.next
    current.next = node_to_insert
    return sorted_head

# if __name__ == "__main__":
ll = LinkedList()
ll.from_list([5, 3, 8, 4, 2])

print("До сортування:")
ll.print()

sorted_head = insertion_sort_linked_list(ll.head)

sorted_ll = LinkedList()
sorted_ll.head = sorted_head

print("Після сортування:")
sorted_ll.print()
