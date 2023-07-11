import numpy as np
import random
import collections

class RRTNode:
    def __init__(self, position, weight=30):
        self.position = position
        self.adjacent_nodes = []
        self.parent = None
        self.weight = weight

class RRT:
    def __init__(self, start, target, layout, max_dist=2):
        self.start = RRTNode(start.astype(int))
        self.target = target
        self.layout = layout
        self.nodes = [self.start]
        self.node_locs = {tuple(self.start.position)}
        self.max_dist = max_dist

        self.open_cells = []
        for y in range(0, layout.height):
            for x in range(0, layout.width):
                if layout.is_open((x, y), known=True) and not layout.is_explored((x, y)):
                    self.open_cells.append((x, y))
        if tuple(self.start.position) in self.open_cells:
            self.open_cells.remove(tuple(self.start.position))
    def update(self, parent):
        for node in parent.adjacent_nodes:
            if self.layout.is_unreachable(node.position):
                self.layout.mark_obstructed(node.position)
                parent.adjacent_nodes.remove(node)
                nearest_node = None
                min_dist = self.max_dist**2
                for sibling in parent.adjacent_nodes:
                    delta = np.array(node.position) - np.array(sibling.position)
                    dist_sq = np.dot(delta, delta)
                    if dist_sq <= min_dist:
                        nearest_node = sibling
                        min_dist = dist_sq
                if nearest_node is None:
                    for child in node.adjacent_nodes:
                        self.remove_node(child)
                else:
                    for child in node.adjacent_nodes:
                        nearest_node.adjacent_nodes.append(child)
                    self.remove_leaf(node)
            elif self.layout.is_explored(tuple(node.position)):
                parent.adjacent_nodes.remove(node)
                for child in node.adjacent_nodes:
                    parent.adjacent_nodes.append(child)
    def has_nearby_node(self, cur_node, agent_pos):
        for node in cur_node.adjacent_nodes[0:3]:
            delta = np.array(agent_pos) - node.position
            if np.dot(delta, delta) <= self.max_dist**2:
                return True
    def plan(self):
        while not self.goal_reached():
            new_node = self.get_new_node()
            self.nodes.append(new_node)
            self.node_locs.add(tuple(new_node.position))
    def get_new_node(self):
        if self.has_valid_open_cell():
            random_point = np.array(random.choice(self.open_cells))
            nearest_node = self.get_nearest_node(random_point)
            delta = random_point - nearest_node.position
            while np.dot(delta, delta) > self.max_dist**2:
                random_point = np.array(random.choice(self.open_cells))
                nearest_node = self.get_nearest_node(random_point)
                delta = random_point - nearest_node.position
            new_node = RRTNode(random_point)
            new_node.parent = nearest_node
            nearest_node.adjacent_nodes.append(new_node)
            self.open_cells.remove(tuple(random_point))
            return new_node
        new_node = RRTNode(self.get_nearest_open_cell())
        new_node.parent = self.start
        if not np.array_equal(new_node.position, self.target):
            self.open_cells.remove(tuple(new_node.position))
        self.start.adjacent_nodes.append(new_node)
        return new_node
    def get_node_by_pos(self, position):
        if position not in self.node_locs:
            return None
        else:
            for node in self.nodes:
                if np.array_equal(node.position, position):
                    return node
    def get_nearest_node(self, position):
        nearest = None
        min_dist = None
        for node in self.nodes:
            dist = np.linalg.norm(node.position - position)
            if nearest is None or dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest
    def get_nearest_open_cell(self):
        if len(self.open_cells) == 0:
            return tuple(self.target)
        seen = set()
        q = collections.deque()
        q.append(tuple(self.start.position))
        while len(q) > 0:
            pos = q.popleft()
            if pos in seen or not self.layout.is_open(pos):
                continue
            seen.add(pos)
            if pos in self.open_cells:
                return pos
            q.append((pos[0] + 1, pos[1]))
            q.append((pos[0], pos[1] + 1))
            q.append((pos[0] - 1, pos[1]))
            q.append((pos[0], pos[1] - 1))
        return tuple(self.target)

        # nearest = self.open_cells[0]
        # min_dist = np.dot(self.start.position - np.array(nearest), self.start.position - np.array(nearest))
        # for cell in self.open_cells:
        #     if cell in self.node_locs:
        #         continue
        #     delta = self.start.position - np.array(cell)
        #     dist_sq = np.dot(delta, delta)
        #     if nearest is None or dist_sq < min_dist:
        #         nearest = cell
        #         min_dist = dist_sq
        # return nearest
    def has_valid_open_cell(self):
        for node_pos in self.node_locs:
            for cell in self.open_cells:
                delta = np.array(node_pos) - cell
                if np.dot(delta, delta) <= self.max_dist**2 and cell not in self.node_locs:
                    return True
        return False
    def goal_reached(self):
        return tuple(self.target) in self.node_locs
    def remove_leaf(self, node):
        self.node_locs.remove(tuple(node.position))
        self.nodes.remove(node)
    def remove_node(self, node):
        while len(node.adjacent_nodes) > 0:
            child = node.adjacent_nodes.pop()
            self.remove_node(child)
        self.remove_leaf(node)
    def mark_explored(self, position):
        for i in range(-1, 2):
            for j in range(-1, 2):
                cell = tuple(np.array(position) + np.array([i, j]))
                if cell in self.open_cells and not np.array_equal(cell, self.target):
                    self.open_cells.remove(cell)
    def print_tree(self):
        print(tuple(self.start.position))
        for node in self.start.adjacent_nodes:
            self.__print_tree(node, '\t')
    def __print_tree(self, node, prefix=''):
        print('{}\___{}'.format(prefix, tuple(node.position)))
        for adj in node.adjacent_nodes:
            self.__print_tree(adj, '{}\t'.format(prefix))