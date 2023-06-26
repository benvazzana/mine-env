import numpy as np
import random

class RRTNode:
    def __init__(self, position, weight=0.1):
        self.position = position
        self.adjacent_nodes = []
        self.parent = None
        self.weight = weight

class RRT:
    def __init__(self, start, target, layout, max_dist=5, open_cells=None):
        self.start = RRTNode(start.astype(int))
        self.target = target
        self.layout = layout
        self.nodes = [self.start]
        self.node_locs = {tuple(self.start.position)}
        self.max_dist = max_dist

        if open_cells is None:
            self.open_cells = []
            for y in range(0, layout.height):
                for x in range(0, layout.width):
                    if layout.is_open((x, y)):
                        self.open_cells.append((x, y))
            self.open_cells.remove(tuple(self.start.position))
        else:
            self.open_cells = open_cells
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
            while tuple(random_point) in self.node_locs or np.dot(delta, delta) > self.max_dist**2:
                random_point = np.array(random.choice(self.open_cells))
                nearest_node = self.get_nearest_node(random_point)
                delta = random_point - nearest_node.position
            new_node = RRTNode(random_point)
            new_node.parent = nearest_node
            nearest_node.adjacent_nodes.append(new_node)
            return new_node
        new_node = RRTNode(self.get_nearest_open_cell())
        new_node.parent = self.start
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
            delta = node.position - position
            dist_sq = np.dot(delta, delta)
            if nearest is None or (self.is_reachable(position, node.position) and dist_sq < min_dist):
                nearest = node
                min_dist = dist_sq
        return nearest
    def get_nearest_open_cell(self):
        nearest = None
        min_dist = None
        for cell in self.open_cells:
            if cell in self.node_locs:
                continue
            delta = self.start.position - np.array(cell)
            dist_sq = np.dot(delta, delta)
            if nearest is None or dist_sq < min_dist:
                nearest = cell
                min_dist = dist_sq
        return nearest
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
        node.parent.adjacent_nodes.remove(node)
        self.nodes.remove(node)
        self.node_locs.remove(tuple(node.position))
    def remove_node(self, node):
        while len(node.adjacent_nodes) > 0:
            child = node.adjacent_nodes[0]
            self.remove_node(child)
        self.remove_leaf(node)
    def mark_explored(self, position):
        for i in range(-1, 2):
            for j in range(-1, 2):
                cell = tuple(np.array(position) + np.array([i, j]))
                if cell in self.open_cells and not np.array_equal(cell, self.target):
                    self.open_cells.remove(cell)
    def is_reachable(self, pos1, pos2):
        # x = min(pos1[0], pos2[0])
        # y = min(pos2[1], pos2[1])
        # width = abs(pos1[0] - pos2[0])
        # height = abs(pos1[1] - pos2[1])
        # free_spaces = 0
        # for i in range(x, x + width + 1):
        #     for j in range(y, y + height + 1):
        #         if self.layout.is_open((i, j)):
        #             free_spaces += 1
        #             if free_spaces >= width + height - 1:
        #                 return True
        return True
    def print_tree(self):
        print(tuple(self.start.position))
        for node in self.start.adjacent_nodes:
            self.__print_tree(node, '\t')
    def __print_tree(self, node, prefix=''):
        print('{}\___{}'.format(prefix, tuple(node.position)))
        for adj in node.adjacent_nodes:
            self.__print_tree(adj, '{}\t'.format(prefix))