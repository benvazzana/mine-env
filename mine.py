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
    
    def render(self, agent_loc, target_loc, mode="human"):
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
        agent_speed = 2 / self.mine_view.framerate
        theta = action[0] - 90
        tolerance = 0.85

        v = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))]) * agent_speed
        new_loc = self.agent_loc + v
        corner1 = np.floor(new_loc + (1 - np.array([1, 1]) * tolerance)).astype(int)
        corner2 = np.floor(new_loc + np.array([1, 0]) * tolerance).astype(int)
        corner3 = np.floor(new_loc + np.array([0, 1]) * tolerance).astype(int)
        corner4 = np.floor(new_loc + np.array([1, 1]) * tolerance).astype(int)
        if self.mine_layout.is_open(corner1) and self.mine_layout.is_open(corner2) and self.mine_layout.is_open(corner3) and self.mine_layout.is_open(corner4):
            self.agent_loc += v
        
        if np.array_equal(corner1, self.target_loc) or np.array_equal(corner2, self.target_loc) or np.array_equal(corner3, self.target_loc) or np.array_equal(corner4, self.target_loc):
            reward = 1
            done = True
        else:
            diff = self.agent_loc - self.target_loc
            reward = -0.005/(self.mine_width * self.mine_height) * np.inner(diff, diff) / self.mine_view.framerate
            done = False
        
        info = {}

        return self.__get_obs(), reward, done, info

    def render(self, mode='human'):
        if mode is not None:
            return self.mine_view.render(self.agent_loc, self.target_loc, mode)
    def reset(self):
        if self.random_targets:
            low = np.array([0, 0])
            high = np.array([self.mine_width - 1, self.mine_height - 1])
            self.target_loc = self.np_random.randint(low, high, dtype=int)
            while not self.mine_layout.is_open(self.target_loc):
                self.target_loc = self.np_random.randint(low, high, dtype=int)
        self.agent_loc = np.zeros(2)
        return self.__get_obs()