from linked_list import LinkedList


class Solution:
    def reverseList(self, head):
        if head is None or head.next is None:
            return head
        new_head = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return new_head

# if __name__ == "__main__":
ll = LinkedList()
ll.from_list([1, 2, 3, 4, 5])

print("До реверсу:")
ll.print()

solver = Solution()
ll.head = solver.reverseList(ll.head)

print("Після реверсу:")
ll.print()


