import numpy as np
import pygame
import os

import gym
from gym import Env
from gym.utils import seeding
from gym.spaces import Box, Discrete

from rrt import RRTNode, RRT

import layouts
import astar
import random


class MineLayout:

    def __init__(self, layout=layouts.LAYOUT0):
        self.layout = layout
        self.__real_layout = np.copy(layout)
        self.known_layout = np.copy(layout)
        self.width = layout.shape[1]
        self.height = layout.shape[0]
        self.cell_shifts = 8

        #print ("self.width, self.height, self.layout", self.width, self.height, self.layout[18][0])

    def __get_nearby_open_cell(self, cell):
        for y in range(-1, 2):
            for x in range(-1, 2):
                if x == 0 and y == 0:
                    continue
                new_cell = cell + np.array([x, y])
                if self.is_open(new_cell):
                    return new_cell
        return None


    def simulate_disaster(self, cell_shifts=8, agent_loc=[(0, 0), (0, 18)]):
        low = np.array([0, 0])
        high = np.array([self.width - 1, self.height - 1])
        # Make random changes to layout
        for i in range(0, cell_shifts):
            cell = np.random.randint(low, high, dtype=int)
            while self.is_open(cell):
                cell = np.random.randint(low, high, dtype=int)
            #print ("cell:", cell)
            open_cell = self.__get_nearby_open_cell(cell)
            #print("open_cell:", open_cell)

            if open_cell is not None:
                for loc in agent_loc:
                    if open_cell is not None and not np.array_equal(loc, open_cell):
                        self.__real_layout[cell[1], cell[0]] = 0
                        self.__real_layout[open_cell[1], open_cell[0]] = 1

    def update(self, position, target):
        explored = False
        for y in range(-1, 2):
            for x in range(-1, 2):
                if position[0] + x < 0 or position[0] + x >= self.width:
                    continue
                if position[1] + y < 0 or position[1] + y >= self.height:
                    continue
                self.known_layout[position[1] + y, position[0] + x] = self.__real_layout[position[1] + y, position[0] + x]
                cell = (position[0] + x, position[1] + y)
                if self.is_open(cell) and not self.is_explored(cell) and not np.array_equal(cell, target):
                    explored = True
                if not np.array_equal(cell, target):
                    self.mark_explored(cell)

                # if not np.array_equal(cell, target):
                #     self.__real_layout[position[1] + y, position[0] + x] = self.known_layout[
                #         position[1] + y, position[0] + x]

        return explored

    def reset(self):
        self.__real_layout = np.copy(self.layout)
        self.known_layout = np.copy(self.layout)

    def mark_obstructed(self, cell):
        self.known_layout[cell[1], cell[0]] = 1

    def is_open(self, cell, known=False):
        if cell[0] >= self.width or cell[1] >= self.height:
            return False
        if cell[0] < 0 or cell[1] < 0:
            return False
        if known:
            return self.known_layout[cell[1], cell[0]] != 1
        else:
            return self.__real_layout[cell[1], cell[0]] != 1
    def is_explored(self, cell):
        if not self.is_open(cell):
            return False
        return self.known_layout[cell[1], cell[0]] >= 2
    def mark_explored(self, cell):
        if not self.is_open(cell):
            return
        if self.get_cell_value(cell, known=True) < 2:
            self.known_layout[cell[1], cell[0]] = 2
        elif self.get_cell_value(cell, known=True) < 10:
            self.known_layout[cell[1], cell[0]] += 0.1
        if self.get_cell_value(cell, known=False) < 2:
            self.__real_layout[cell[1], cell[0]] = 2
        elif self.get_cell_value(cell, known=False) < 10:
            self.__real_layout[cell[1], cell[0]] += 0.1
    def get_cell_value(self, cell, known=False):
        if known:
            return self.known_layout[cell[1], cell[0]]
        else:
            return self.__real_layout[cell[1], cell[0]]
    def is_unreachable(self, cell, known=False):
        if not self.is_open(cell, known):
            return True
        obstructed = True
        if cell[0] > 0 and self.is_open(cell + np.array([-1, 0]), known):
            obstructed = False
        if cell[0] < self.width - 1 and self.is_open(cell + np.array([1, 0]), known):
            obstructed = False
        if cell[1] > 0 and self.is_open(cell + np.array([0, -1]), known):
            obstructed = False
        if cell[1] > self.height - 1 and self.is_open(cell + np.array([0, 1]), known):
            obstructed = False
        return obstructed

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
# 2, 4, 5
    def render(self, agent_num, agent_loc, target_loc, rrt_nodes=None, rrt=None, mode="human"):
        canvas = pygame.Surface(self.screen_size).convert_alpha()
        canvas.fill((0, 0, 0, 0))
        cell_width = self.screen_size[0] / self.layout.width
        cell_height = self.screen_size[1] / self.layout.height
        for y in range(0, self.layout.height):
            for x in range(0, self.layout.width):
                color = None
                if not self.layout.is_real_cell((x, y)) and pygame.key.get_pressed()[pygame.K_LSHIFT]:
                    if self.layout.is_open((x, y)):
                        color = (0, 100, 255, 50)
                    else:
                        color = (255, 0, 0, 100)
                elif not self.layout.is_open((x, y), known=True):
                    color = (0, 0, 0, 255)
                elif self.layout.is_explored((x, y)):
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

        for num in range(agent_num):
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                ((agent_loc[num] + 0.5) * np.array([cell_width, cell_height])).astype(int),
                cell_width / 3,
            )
            if rrt_nodes[num] is not None:
                rrt_node_color = (255, 255, 255)
                for node in rrt_nodes[num]:
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

    SPEED = 3
    action_to_direction = {
        0: np.array([0, -1]),
        1: np.array([-1, 0]),
        2: np.array([0, 1]),
        3: np.array([1, 0])
    }

    def __init__(self, mine_layout=None, target_loc=None, framerate=60, screen_size=(640, 640)):
        self.agent_loc = None
        self.agent_num = 4
        if mine_layout is None:
            self.mine_layout = MineLayout()
        else:
            self.mine_layout = mine_layout
        self.mine_view = MineView(mine_layout=self.mine_layout, framerate=framerate, screen_size=screen_size)
        self.metadata['render_fps'] = self.mine_view.framerate
        self.seed()

        self.mine_width = self.mine_layout.width
        self.mine_height = self.mine_layout.height

        # Change in direction of travel [-10, +10]
        # self.action_space = Discrete(21)
        self.action_space = Discrete(4)

        # State: agent position, 3 RRT node positions (including weights), target position,
        # angle, surrounding obstacles
        self.obs_size = 2 + 2 + (3 * 3) + (5 * 5)
        #self.obs_size2 = 2 + 2 + (3 * 3) + 1 + (5 * 5)


        # For floating point rounding
        low = np.zeros(self.obs_size)
        high = np.zeros(self.obs_size)
        high[[0, 2, 4, 7, 10]] = self.mine_width - 1
        high[[1, 3, 5, 8, 11]] = self.mine_height - 1
        high[[6, 9, 12]] = 100  # RRT node weights
        high[13:] = 10
        self.observation_space = Box(low, high, dtype=np.float64)

        self.target_loc = target_loc
        if self.target_loc is None:
            self.random_targets = True
        else:
            self.random_targets = False

    def __get_obs(self):
        observation = [[], [], [], []]
        target_nodes = []
        for i in range(self.agent_num):
            observation[i] = np.zeros(self.obs_size).astype(np.float64)
            #print (observation[i][[0, 1]], self.agent_loc[i])
            #print(observation)
            observation[i][[0, 1]] = self.agent_loc[i]
            #print(observation[0])
            observation[i][[2, 3]] = self.target_loc
            if len(self.cur_path[i]) > 0:
                target_nodes.append(RRTNode(self.cur_path[i][0], self.cur_rrt_node[i].weight / 2))
            for node in self.cur_rrt_node[i].adjacent_nodes:
                target_nodes.append(node)
                if len(target_nodes) == 3:
                    break
            while len(target_nodes) < 3:
                target_nodes.append(RRTNode(self.target_loc, 0))
            for x in range(3):
                observation[i][[4 + (3 * x), 5 + (3 * x)]] = target_nodes[x].position
                observation[i][6 + 3 * x] = target_nodes[x].weight
            surrounding_cells = np.zeros(5 * 5).astype(np.float64) + 1
            j = 0
            for y in range(-2, 3):
                for x in range(-2, 3):
                    cell = (self.agent_loc[i] + [0.5, 0.5]).astype(int) + np.array([x, y])
                    # print('Cell: {} = {}'.format(cell, self.mine_layout.get_cell_value(cell)))
                    if self.mine_layout.is_open(cell, known=True):
                        surrounding_cells[j] = self.mine_layout.get_cell_value(cell, known=True)
                    j += 1

            observation[i][13:] = surrounding_cells
        #print ("observations", observation)
        return observation


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        rewards = []
        dones = []

        for i in range(self.agent_num):
            done = False
            reward = 0
            wall_collision = False
            #print ("ACTIONS IN STEP",actions[0])
            new_loc = self.agent_loc[i] + self.action_to_direction[actions[i]]
            if self.mine_layout.is_open(new_loc.astype(int)):
                self.agent_loc[i] = new_loc
            else:
                wall_collision = True
            cell_pos = tuple(self.agent_loc[i].astype(int))

            target_found = False
            explored = False
            path_traversed = False
            if np.array_equal(self.agent_loc[i], self.target_loc):
                target_found = True
            exploration_nodes = self.cur_rrt_node[i].adjacent_nodes
            if len(exploration_nodes) > 3:
                exploration_nodes = exploration_nodes[0:3]
            for node in exploration_nodes:
                if np.array_equal(node.position, self.agent_loc[i]):
                    explored = True
                    self.cur_rrt_node[i] = node
            if len(self.cur_path[i]) > 0:
                if np.array_equal(self.cur_path[i][0], cell_pos):
                    path_traversed = True
                    self.cur_path[i].pop(0)

            self.rrt[i].update(self.cur_rrt_node[i])
            # If a leaf is reached, regenerate tree
            has_nearby_node = self.rrt[i].has_nearby_node(self.cur_rrt_node[i], self.agent_loc[i])
            if not target_found:
                if len(self.cur_rrt_node[i].adjacent_nodes) == 0:
                    self.rrt[i] = RRT(self.agent_loc[i], self.target_loc, self.mine_layout)
                    self.cur_rrt_node[i] = self.rrt[i].nodes[0]
                    self.rrt[i].plan()
                if not has_nearby_node:
                    if len(self.cur_path[i]) == 0:
                        path = astar.bfs(self.mine_layout, cell_pos)
                        if path is not None:
                            path.pop(0)
                            self.cur_path[i] = path
            dist = np.linalg.norm(self.agent_loc[i] - self.target_loc)

            if target_found:
                reward = 200
                done = True
            elif explored:
                reward = self.cur_rrt_node[i].weight / dist
                self.cur_path[i] = []
            elif path_traversed:
                reward = self.cur_rrt_node[i].weight / (2 * dist)
            else:
                reward = -0.1 * self.mine_layout.get_cell_value(cell_pos)

            explored = self.mine_layout.update(tuple(self.agent_loc[i].astype(int)), self.target_loc)
            if explored:
                # reward = 1
                if not has_nearby_node:
                    self.rrt[i] = RRT(self.agent_loc[i], self.target_loc, self.mine_layout)
                    self.cur_rrt_node[i] = self.rrt[i].nodes[0]
                    self.rrt[i].plan()
            if not self.mine_layout.is_open(self.target_loc, known=True):
                print('An episode was terminated because the agent was trapped')
                done = True
            rewards.append(reward)
            dones.append(done)

        info = {}

        return self.__get_obs(), rewards, dones, info

    def render(self, mode='human'):
        if mode is not None:
            cur_rrt_node = []
            multi_adjacent_nodes = []

            for i in range(self.agent_num):
                adjacent_nodes = []
                if len(self.cur_path[i]) > 0:
                    adjacent_nodes.append(RRTNode(self.cur_path[i][0], self.cur_rrt_node[i].weight))
                for node in self.cur_rrt_node[i].adjacent_nodes[0:3 - len(adjacent_nodes)]:
                    adjacent_nodes.append(node)
                multi_adjacent_nodes.append(adjacent_nodes)


            # for i in range(self.agent_num):
            #     cur_rrt_node.append(self.cur_rrt_node[i].adjacent_nodes[0:3])
            return self.mine_view.render(self.agent_num, self.agent_loc, self.target_loc, multi_adjacent_nodes,
                                         rrt=self.rrt, mode=mode)

    def reset(self):
        self.mine_layout.reset()
        valid_positions = []
        self.agent_index = []
        self.agent_loc = []


        for j in range(len(self.mine_layout.layout)):
            for x in range(j):
                if self.mine_layout.layout[x][j] == 0:
                    valid_positions.append([j, x])

        for i in range(self.agent_num):
            # valid_positions = np.where(self.mine_layout == 0)

            same_loc = True
            agent_loc = random.choice(valid_positions)
            while same_loc:

                agent_loc = random.choice(valid_positions)
                if len(self.agent_index) == 0:
                    self.agent_index.append(agent_loc)
                    break
                elif agent_loc not in self.agent_index:
                    self.agent_index.append(agent_loc)
                    break
            self.agent_loc.append(np.array([agent_loc[0], agent_loc[1]]))


        self.mine_layout.simulate_disaster(agent_loc= self.agent_index)
        if self.random_targets:
            low = np.array([0, 0])
            high = np.array([self.mine_width - 1, self.mine_height - 1])
            self.target_loc = self.np_random.randint(low, high, dtype=int)
            while not self.mine_layout.is_open(self.target_loc):
                self.target_loc = self.np_random.randint(low, high, dtype=int)


        self.angle = []
        self.rrt = []
        self.cur_rrt_node = []
        self.cur_path = []







        for i in range (self.agent_num):
            #valid_positions = np.where(self.mine_layout == 0)


            same_loc = True
            agent_loc = random.choice(valid_positions)
            while same_loc:

                agent_loc = random.choice(valid_positions)
                if len(self.agent_index) == 0:
                    self.agent_index.append(agent_loc)
                    break
                elif agent_loc not in self.agent_index:
                    self.agent_index.append(agent_loc)
                    break


            self.angle.append(30)
            #self.agent_loc.append(np.array([i, 0.0]))

            print (agent_loc)
            self.agent_loc.append(np.array([agent_loc[0], agent_loc[1]]))

            #print(self.agent_loc)
            self.rrt.append(RRT(self.agent_loc[i], self.target_loc, self.mine_layout))
            self.cur_rrt_node.append(self.rrt[i].nodes[0])
            self.rrt[i].plan()
            self.cur_path.append([])

        #self.agent_loc.append(np.zeros(2))

        return self.__get_obs()