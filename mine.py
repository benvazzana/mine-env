import numpy as np
import pygame

import gym
from gym import Env
from gym.utils import seeding
from gym.spaces import Box, Discrete

import layouts
import random

class MineLayout:

    def __init__(self, layout=layouts.LAYOUT0):
        self.layout = layout
        self.width = layout.shape[1]
        self.height = layout.shape[0]
    
    def is_open(self, cell):
        if cell[0] >= self.width or cell[1] >= self.height:
            return False
        if cell[0] < 0 or cell[1] < 0:
            return False
        return self.layout[cell[1], cell[0]] == 0

class MineView:

    def __init__(self, mine_layout=None, framerate=4, screen_size=(640, 640), name="OpenAI Gym: Mine Environment"):
        if mine_layout is None:
            self.layout = MineLayout()
        else:
            self.layout = mine_layout
        
        pygame.init()
        pygame.display.set_caption(name)
        
        self.framerate = framerate
        self.screen_size = screen_size
        self.window = pygame.display.set_mode(screen_size)
        self.clock = pygame.time.Clock()

        self.background = pygame.Surface(screen_size).convert()
        self.background.fill((150, 150, 150))
    
    def render(self, agent_loc, target_loc, rrt_nodes=None, mode="human"):
        canvas = pygame.Surface(self.screen_size).convert_alpha()
        canvas.fill((0, 0, 0, 0))
        cell_width = self.screen_size[0] / self.layout.width
        cell_height = self.screen_size[1] / self.layout.height
        for y in range(0, self.layout.height):
            for x in range(0, self.layout.width):
                color = None
                if not self.layout.is_open((x, y)):
                    color = (0, 0, 0)
                #if np.array_equal(agent_loc, np.array([x, y])):
                #    color = (0, 0, 255)
                if np.array_equal(target_loc, np.array([x, y])):
                    color = (255, 255, 0)
                
                if color is not None:
                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(
                            np.array([x * cell_width, y * cell_height]),
                            (cell_width, cell_height)
                        )
                    )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            ((agent_loc + 0.5) * np.array([cell_width, cell_height])).astype(int),
            cell_width / 3,
        )
        if rrt_nodes is not None:
            rrt_node_color = (255, 255, 255)
            for node in rrt_nodes:
                x, y = node.position
                pygame.draw.rect(
                    canvas,
                    rrt_node_color,
                    pygame.Rect(
                        np.array([x * cell_width, y * cell_height]),
                        (cell_width, cell_height)
                    )
                )

        self.window.blit(self.background, (0, 0))
        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.framerate)
        
        return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

class RRTNode:
    def __init__(self, position):
        self.position = position
        self.adjacent_nodes = []
        self.parent = None

class RRT:
    def __init__(self, start, target, layout, max_dist=5):
        self.start = RRTNode(start)
        self.target = RRTNode(target)
        self.layout = layout
        self.open_cells = []
        self.nodes = [self.start]
        self.node_locs = {tuple(self.start.position)}
        self.max_dist = max_dist

        for y in range(0, layout.height):
            for x in range(0, layout.width):
                if layout.is_open((x, y)):
                    self.open_cells.append((x, y))
        self.open_cells.remove(tuple(self.start.position))
    def plan(self):
        while not self.goal_reached():
            new_node = self.get_new_node()
            self.nodes.append(new_node)
            self.node_locs.add(tuple(new_node.position))
            self.open_cells.remove(tuple(new_node.position))
    def get_new_node(self):
        random_point = np.array(random.choice(self.open_cells))
        nearest_node = self.get_nearest_node(random_point)
        dx = random_point - nearest_node.position
        while np.dot(dx, dx) > self.max_dist**2:
            random_point = np.array(random.choice(self.open_cells))
            nearest_node = self.get_nearest_node(random_point)
            dx = random_point - nearest_node.position
        new_node = RRTNode(random_point)
        new_node.parent = nearest_node
        nearest_node.adjacent_nodes.append(new_node)
        return new_node
    def get_node_by_pos(self, position):
        if position not in self.node_locs:
            return None
        else:
            for node in self.nodes:
                if np.array_equal(node.position, position):
                    return node
    def get_nearest_node(self, position):
        return min(self.nodes, key=lambda node: np.dot(node.position - position, node.position - position))
    def goal_reached(self):
        return tuple(self.target.position) in self.node_locs
    def remove_leaf(self, node):
        node.parent.adjacent_nodes.remove(node)
    def print_tree(self):
        print(tuple(self.start.position))
        for node in self.start.adjacent_nodes:
            self.__print_tree(node, '\t')
    def __print_tree(self, node, prefix=''):
        print('{}\___{}'.format(prefix, tuple(node.position)))
        for adj in node.adjacent_nodes:
            self.__print_tree(adj, '{}\t'.format(prefix))

class MineEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, mine_layout=None, target_loc=None, framerate=60, screen_size=(640, 640)):
        if mine_layout is None:
            self.mine_layout = MineLayout()
            self.mine_view = MineView(mine_layout=self.mine_layout, framerate=framerate, screen_size=screen_size)
        else:
            self.mine_layout = mine_layout
            self.mine_view = MineView(mine_layout=self.mine_layout, framerate=framerate, screen_size=screen_size)
        self.metadata['render_fps'] = self.mine_view.framerate
        self.seed()

        self.mine_width = self.mine_layout.width
        self.mine_height = self.mine_layout.height

        # Direction of travel
        self.action_space = Box(0, 359, shape=(1,))

        # State: agent position, target position, flattened layout of obstacles
        self.obs_size = 2 + 2 + (self.mine_width * self.mine_height)
        low = np.zeros(self.obs_size)
        high = np.zeros(self.obs_size)
        high[[0, 2]] = self.mine_width - 1
        high[[1, 3]] = self.mine_height - 1
        high[4:] = 1
        self.observation_space = Box(low, high, dtype=int)

        self.target_loc = target_loc
        if self.target_loc is None:
            self.random_targets = True
        else:
            self.random_targets = False
    def __get_obs(self):
        observation = np.zeros(self.obs_size).astype(int)
        observation[[0, 1]] = self.agent_loc.astype(int)
        observation[[2, 3]] = self.target_loc
        observation[4:] = self.mine_view.layout.layout.flatten()
        return observation
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):
        done = False
        agent_speed = 2 / self.mine_view.framerate
        theta = action[0] - 90
        TOLERANCE = 0.85

        v = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))]) * agent_speed
        new_loc = self.agent_loc + v
        agent_vertices = [
            np.floor(new_loc + (1 - (np.array([1, 1]) * TOLERANCE))).astype(int),
            np.floor(new_loc + np.array([TOLERANCE, 1 - TOLERANCE])).astype(int),
            np.floor(new_loc + np.array([1 - TOLERANCE, TOLERANCE])).astype(int),
            np.floor(new_loc + np.array([1, 1]) * TOLERANCE).astype(int)
        ]

        wall_collision = False
        for vertex in agent_vertices:
            if not self.mine_layout.is_open(vertex):
                wall_collision = True
        if not wall_collision:
            self.agent_loc += v

        target_found = False
        for vertex in agent_vertices:
            if np.array_equal(vertex, self.target_loc):
                target_found = True

        # Check if agent has reached RRT exploration node
        explored = False
        for vertex in agent_vertices:
            exploration_nodes = self.cur_rrt_node.adjacent_nodes
            if len(exploration_nodes) > 3:
                exploration_nodes = exploration_nodes[0:3]
            for node in exploration_nodes:
                if np.array_equal(node.position, vertex):
                    explored = True
                    self.cur_rrt_node = node

        # Remove leaf nodes from past trajectory until new exploration node is found
        while len(self.cur_rrt_node.adjacent_nodes) == 0:
            self.rrt.remove_leaf(self.cur_rrt_node)
            self.cur_rrt_node = self.cur_rrt_node.parent

        if target_found:
            reward = 1
            done = True
        elif explored:
            reward = 0.1
        else:
            reward = -0.1/(self.mine_width * self.mine_height * self.mine_view.framerate)
        
        info = {}

        return self.__get_obs(), reward, done, info

    def render(self, mode='human'):
        if mode is not None:
            return self.mine_view.render(self.agent_loc, self.target_loc, self.cur_rrt_node.adjacent_nodes, mode=mode)
    def reset(self):
        if self.random_targets:
            low = np.array([0, 0])
            high = np.array([self.mine_width - 1, self.mine_height - 1])
            self.target_loc = self.np_random.randint(low, high, dtype=int)
            while not self.mine_layout.is_open(self.target_loc):
                self.target_loc = self.np_random.randint(low, high, dtype=int)
        self.agent_loc = np.zeros(2)
        self.rrt = RRT(self.agent_loc, self.target_loc, self.mine_layout)
        self.cur_rrt_node = self.rrt.nodes[0]
        self.rrt.plan()
        return self.__get_obs()