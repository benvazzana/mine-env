import numpy as np
import pygame

import gym
from gym import Env
from gym.utils import seeding
from gym.spaces import Box, Discrete

layout0 = np.array([
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0]
])

class MineLayout:

    COMPASS = {
        'N': np.array([0, -1]),
        'E': np.array([1, 0]),
        'S': np.array([0, 1]),
        'W': np.array([-1, 0]),
    }

    def __init__(self, layout=layout0):
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
                if np.array_equal(agent_loc, np.array([x, y])):
                    color = (0, 0, 255)
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

    def __init__(self, mine_layout=None, target_loc=None, screen_size=(640, 640)):
        if mine_layout is None:
            self.mine_layout = MineLayout()
            self.mine_view = MineView(mine_layout=self.mine_layout, screen_size=screen_size)
        else:
            self.mine_layout = mine_layout
            self.mine_view = MineView(mine_layout=self.mine_layout, screen_size=screen_size)
        self.metadata['render_fps'] = self.mine_view.framerate
        self.seed()

        self.mine_width = self.mine_layout.width
        self.mine_height = self.mine_layout.height

        # Up, down, left, right
        self.action_space = Discrete(4)

        low = np.array([0, 0])
        high = np.array([self.mine_width - 1, self.mine_height - 1])
        # Agent position
        self.observation_space = Box(low, high, dtype=int)

        self.target_loc = target_loc
        if self.target_loc is None:
            self.target_loc = self.np_random.randint(low, high, dtype=int)
            while not self.mine_layout.is_open(self.target_loc):
                self.target_loc = self.np_random.randint(low, high, dtype=int)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):
        direction = self.ACTION[action]

        new_loc = self.state + self.mine_layout.COMPASS[direction]
        if self.mine_layout.is_open(new_loc):
            self.state = new_loc
        
        if np.array_equal(self.state, self.target_loc):
            reward = 1
            done = True
        else:
            reward = -0.1/(self.mine_width* self.mine_height)
            done = False
        
        info = {}

        return self.state, reward, done, info

    def render(self, mode='human'):
        if mode is not None:
            return self.mine_view.render(self.state, self.target_loc, mode)
    def reset(self):
        self.state = np.zeros(2).astype(int)
        return self.state