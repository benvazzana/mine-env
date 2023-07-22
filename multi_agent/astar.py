import collections


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        return self.position == other.position


def bfs(layout, start):
    start_node = Node(None, tuple(start))
    seen = set()

    q = collections.deque()
    q.append(start_node)
    while len(q) > 0:
        node = q.popleft()
        if node.position in seen or not layout.is_open(node.position):
            continue
        seen.add(node.position)
        if not layout.is_explored(node.position):
            path = []
            current = node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]
        for pos_change in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
            new_pos = (node.position[0] + pos_change[0], node.position[1] + pos_change[1])
            q.append(Node(node, new_pos))
    return None


def shortest_path(layout, start, end, known=True):
    start_node = Node(None, tuple(start))
    end_node = Node(None, tuple(end))

    result = []
    open_list = []
    closed_list = set()

    open_list.append(start_node)

    while len(open_list) > 0:
        current_node = open_list[0]
        result.append(current_node.position)

        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        open_list.pop(current_index)
        closed_list.add(current_node.position)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            within_range = [
                0 <= node_position[0] < layout.width,
                0 <= node_position[1] < layout.height,
            ]

            if not all(within_range):
                continue

            if not layout.is_open(node_position, known=known):
                continue

            new_node = Node(current_node, node_position)
            children.append(new_node)

        for child in children:
            if child.position in closed_list:
                continue

            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                        (child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            open_list.append(child)
    # print('Failed pathfind from {} to {}'.format(start, end))
    # l = layout.get_layout()
    # for row in l:
    #     print(row.astype(int))