import numpy as np
import pygame

import gym
from gym import Env
from gym.utils import seeding
from gym.spaces import Box, Discrete

import layouts

class MineLayout:

    COMPASS = {
        'N': np.array([0, -1]),
        'E': np.array([1, 0]),
        'S': np.array([0, 1]),
        'W': np.array([-1, 0]),
    }

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
    
    def render(self, agent_loc, target_loc, mode="human"):
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
                if np.array_equal(agent_loc, np.array([x, y])):
                    color = (0, 0, 255, 255)
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

    ACTION = ['N', 'E', 'S', 'W']

    def __init__(self, mine_layout=None, target_loc=None, framerate=4, screen_size=(640, 640)):
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

        # Up, down, left, right
        self.action_space = Discrete(4)

        # Observation space: Agent pos relative to target, pre-disaster mine layout
        self.obs_size = 2 + (self.mine_width * self.mine_height)
        bound = np.zeros(self.obs_size)
        bound[0] = self.mine_width - 1
        bound[1] = self.mine_height - 1
        bound[2:] = 1
        self.observation_space = Box(-bound, bound, dtype=int)

        self.target_loc = target_loc
        if self.target_loc is None:
            self.random_targets = True
        else:
            self.random_targets = False
    def __get_obs(self):
        observation = np.zeros(self.obs_size).astype(int)
        observation[[0, 1]] = self.agent_loc - self.target_loc
        observation[2:] = self.mine_view.layout.get_layout().flatten()
        return observation
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):
        direction = self.ACTION[action]

        new_loc = self.agent_loc + self.mine_layout.COMPASS[direction]
        if self.mine_layout.is_open(new_loc):
            self.agent_loc = new_loc
        if np.array_equal(self.agent_loc, self.target_loc):
            reward = 1
            done = True
        else:
            rel_pos = self.agent_loc - self.target_loc
            reward = -0.01*(np.inner(rel_pos, rel_pos)/(self.mine_width*self.mine_height))
            done = False
        
        info = {}

        return self.__get_obs(), reward, done, info

    def render(self, mode='human'):
        if mode is not None:
            return self.mine_view.render(self.agent_loc, self.target_loc, mode)
    def reset(self):
        self.mine_layout.reset()
        self.mine_layout.simulate_disaster()
        if self.random_targets:
            low = np.array([0, 0])
            high = np.array([self.mine_width - 1, self.mine_height - 1])
            self.target_loc = self.np_random.randint(low, high, dtype=int)
            while not self.mine_layout.is_open(self.target_loc):
                self.target_loc = self.np_random.randint(low, high, dtype=int)
        self.agent_loc = np.zeros(2).astype(int)
        return self.__get_obs()