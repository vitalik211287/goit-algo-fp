from linked_list import LinkedList, Node


def merge_sorted_lists(head1, head2):
    dummy = Node()
    tail = dummy

    while head1 and head2:
        if head1.data < head2.data:
            tail.next = head1
            head1 = head1.next
        else:
            tail.next = head2
            head2 = head2.next
        tail = tail.next

    # Додаємо залишки одного з списків
    if head1:
        tail.next = head1
    elif head2:
        tail.next = head2

    return dummy.next

# Створимо два відсортовані списки
l1 = LinkedList()
l1.from_list([1, 3, 5])
print("Список 1:")
l1.print()

l2 = LinkedList()
l2.from_list([2, 4, 6])
print("Список 2:")
l2.print()

# Об'єднання
merged_head = merge_sorted_lists(l1.head, l2.head)
print("Об'єднаний список:")
merged = LinkedList()
merged.head = merged_head
merged.print()

