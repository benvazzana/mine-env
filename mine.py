import numpy as np
import pygame

import gym
from gym import Env
from gym.utils import seeding
from gym.spaces import Box, Discrete

from rrt import RRTNode, RRT

import layouts
import random

class MineLayout:

    def __init__(self, layout=layouts.LAYOUT0):
        self.__real_layout = layout
        self.known_layout = np.copy(layout)
        self.width = layout.shape[1]
        self.height = layout.shape[0]
    
    def __get_nearby_open_cell(self, cell):
        for y in range(-1, 2):
            for x in range(-1, 2):
                if x == 0 and y == 0:
                    continue
                new_cell = cell + np.array([x, y])
                if self.is_open(new_cell):
                    return new_cell
        return None
    
    def simulate_disaster(self, cell_shifts=8):
        low = np.array([0, 0])
        high = np.array([self.width - 1, self.height - 1])
        # Make random changes to layout
        for i in range(0, cell_shifts):
            cell = np.random.randint(low, high, dtype=int)
            while self.is_open(cell):
                cell = np.random.randint(low, high, dtype=int)
            open_cell = self.__get_nearby_open_cell(cell)
            if open_cell is not None:
                self.__real_layout[cell[1], cell[0]] = 0
                self.__real_layout[open_cell[1], open_cell[0]] = 1
    
    def reset(self):
        self.__real_layout = np.copy(self.known_layout)
    
    def is_open(self, cell):
        if cell[0] >= self.width or cell[1] >= self.height:
            return False
        if cell[0] < 0 or cell[1] < 0:
            return False
        return self.__real_layout[cell[1], cell[0]] == 0
    def is_real_cell(self, cell):
        return self.__real_layout[cell[1], cell[0]] == self.known_layout[cell[1], cell[0]]
    def get_layout(self):
        return self.known_layout

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
    
    def render(self, agent_loc, target_loc, rrt_nodes=None, rrt=None, mode="human"):
        canvas = pygame.Surface(self.screen_size).convert_alpha()
        canvas.fill((0, 0, 0, 0))
        cell_width = self.screen_size[0] / self.layout.width
        cell_height = self.screen_size[1] / self.layout.height
        for y in range(0, self.layout.height):
            for x in range(0, self.layout.width):
                color = None
                if not self.layout.is_real_cell((x, y)) and pygame.key.get_pressed()[pygame.K_LSHIFT]:
                    if self.layout.is_open((x, y)):
                        color = (0, 0, 0, 50)
                    else:
                        color = (255, 0, 0, 100)
                elif not self.layout.is_open((x, y)):
                    color = (0, 0, 0, 255)
                elif rrt is not None:
                    if (x, y) not in rrt.open_cells:
                        color = (200, 200, 200)
                if np.array_equal(target_loc, np.array([x, y])):
                    color = (255, 255, 0, 255)
                
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
        self.action_space = Discrete(360)

        # State: agent position, 3 RRT node positions, target position, flattened layout of obstacles
        self.obs_size = 2 + 2 + (2 * 3) + (self.mine_width * self.mine_height)
        low = np.zeros(self.obs_size)
        high = np.zeros(self.obs_size)
        high[0:10:2] = self.mine_width - 1
        high[1:10:2] = self.mine_height - 1
        high[10:] = 1
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
        rrt_nodes = [node.position for node in self.cur_rrt_node.adjacent_nodes[0:3]]
        while len(rrt_nodes) < 3:
            rrt_nodes.append(self.target_loc)
        for i in range(0, 3):
            observation[[4 + (2 * i), 5 + (2 * i)]] = rrt_nodes[i]
        observation[10:] = self.mine_layout.known_layout.flatten()
        return observation
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):
        done = False
        agent_speed = 10 / self.mine_view.framerate
        theta = action - 90
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
        rrt_pos = tuple((self.agent_loc + np.array([0.5, 0.5])).astype(int))
        self.rrt.mark_explored(rrt_pos)

        # If a leaf is reached, regenerate tree
        if not target_found and len(self.cur_rrt_node.adjacent_nodes) == 0:
            open_cells = self.rrt.open_cells
            self.rrt = RRT(self.agent_loc, self.target_loc, self.mine_layout, open_cells=open_cells)
            self.cur_rrt_node = self.rrt.nodes[0]
            self.rrt.plan()

        if target_found:
            reward = 1
            done = True
        elif explored:
            delta = self.agent_loc - self.target_loc
            dist_sq = np.dot(delta, delta)
            reward = 1 / dist_sq
        else:
            reward = agent_speed * -0.1/(self.mine_width * self.mine_height)
        
        info = {}

        return self.__get_obs(), reward, done, info

    def render(self, mode='human'):
        if mode is not None:
            return self.mine_view.render(self.agent_loc, self.target_loc, self.cur_rrt_node.adjacent_nodes[0:3], rrt=self.rrt, mode=mode)
    def reset(self):
        self.mine_layout.reset()
        self.mine_layout.simulate_disaster()
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