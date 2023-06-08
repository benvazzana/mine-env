import gym
from gym import spaces
import numpy as np
import pygame
from random import randrange
import time

# Pygame colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
CYAN = (0, 255, 255)

# Environment configuration
GRID_SIZE = 10
WIN_SIZE = 600

# Map indices
UNEXPLORED_INDEX = -1
EMPTY_INDEX = 0
AGENT1_INDEX = 1
AGENT2_INDEX = 2
GOAL_INDEX = 3
OBSTACLE_INDEX = 4
WALL_INDEX = 5

# Define actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_DIRECTIONS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

actions_list = [UP, DOWN, LEFT, RIGHT]


class MyCustomEnv(gym.Env):
    def __init__(self):
        super(MyCustomEnv, self).__init__()

        self._done2 = False
        self._done1 = False
        self.action_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4)))  # Four directions for each agent
        self.observation_space = spaces.Box(low=-1, high=5, shape=(GRID_SIZE, GRID_SIZE),
                                            dtype=np.int8)  # Global observation

        # Initialize Pygame
        pygame.init()
        self.win = pygame.display.set_mode((WIN_SIZE, WIN_SIZE))
        self.cell_size = WIN_SIZE // GRID_SIZE
        self.astar_path_1 = []
        self.astar_path_2 = []

        # Initialize shared map
        self.shared_map = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ], dtype=np.int8)

        # Initialize fixed map layout
        self.fixed_map = np.array([
            [5, 0, 0, 0, 5, 5, 0, 0, 0, 5],
            [5, 5, 5, 0, 0, 5, 5, 5, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 5, 1, 3],
            [0, 0, 5, 5, 5, 5, 0, 0, 0, 5],
            [0, 0, 5, 0, 0, 5, 5, 0, 4, 0],
            [5, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 5, 4, 3],
            [5, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [5, 5, 5, 5, 0, 0, 0, 0, 0, 5]
        ], dtype=np.int8)

        self.border = int((len(self.fixed_map) / 2)) - 1
        self.goal = None
        # print(self.border)

        self.agent_positions = []
        self.goal_positions = []
        self.obstacle_positions = []
        self.wall_positions = []

        self.agent1_pos = np.array(np.where(self.fixed_map == AGENT1_INDEX)).reshape(-1)
        self.agent2_pos = np.array(np.where(self.fixed_map == AGENT2_INDEX)).reshape(-1)

        self.agent1_done = False
        self.agent2_done = False

        self.agent1_inform = False
        self.agent2_inform = False

        self.meet = False

    def reset(self):
        # Reset shared map
        self.shared_map = np.full((GRID_SIZE, GRID_SIZE), UNEXPLORED_INDEX, dtype=np.int8)
        # Update agents' positions on shared map
        self.shared_map[self.agent1_pos[0], self.agent1_pos[1]] = AGENT1_INDEX
        self.shared_map[self.agent2_pos[0], self.agent2_pos[1]] = AGENT2_INDEX
        # self._update_agent_view(self.fixed_map)

        self.agent1_done = False
        self.agent2_done = False

        self.agent1_inform = False
        self.agent2_inform = False

        self.meet = False

        self.goal = None
        self.astar_path_1 = []
        self.astar_path_2 = []

        return self._get_observation()

    def _in_bounds(self, pos, agent_number):
        """Check if a position is within the grid boundaries."""
        if agent_number == AGENT1_INDEX and self.agent1_done is False:
            return 0 <= pos[0] <= self.border and 0 <= pos[1] < GRID_SIZE
        if agent_number == AGENT2_INDEX and self.agent2_done is False:
            return self.border < pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE

        return 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE


    def _get_action_towards_goal(self, pos, agent_number):
        for action, delta in ACTION_DIRECTIONS.items():
            new_pos = pos + np.array(delta)
            if self._in_bounds(new_pos, agent_number):
                if self.shared_map[new_pos[0], new_pos[1]] == GOAL_INDEX:
                    self.goal = (new_pos[0], new_pos[1])
                    return -1  # move towards the exit if it's in sight
                elif self.shared_map[new_pos[0], new_pos[1]] == UNEXPLORED_INDEX:
                    # print(action)
                    return action  # move towards unexplored areas if there's no exit in sight

                elif self.shared_map[new_pos[0], new_pos[1]] == AGENT1_INDEX or self.shared_map[new_pos[0], new_pos[1]] == AGENT2_INDEX:
                    # print(action)
                    if self.goal is not None:
                        self.meet = True
                        return -2
                    else:
                        return

                # self.meet = False

        return -1  # return a random action if there are no unexplored areas or exits in sight

    def step(self, actions):
        _done = False

        # Update positions if new positions are empty
        if not self.agent1_done:
            if self.meet is True and self.goal is not None:
                if not self.astar_path_1:
                    # print(self.agent2_pos, self.shared_map[2, 9])
                    self.astar_path_1 = astar(self.fixed_map, (self.agent1_pos[0], self.agent1_pos[1]), self.goal)
                new_pos1 = self.astar_path_1.pop(0)  # Take the next step in the path

                if new_pos1 is not None:
                    best_action1 = self._get_direction(self.agent1_pos, new_pos1)
                    self._perform_action(best_action1, self.agent1_pos, AGENT1_INDEX)
                    print(self.agent1_pos, new_pos1)
            else:
                best_action1 = self._get_action_towards_goal(self.agent1_pos, AGENT1_INDEX)
                if best_action1 == -1:
                    temp_action = self.action_space.sample()
                    best_action1 = temp_action[1]
                # print(self.agent1_done)
                self._perform_action(best_action1, self.agent1_pos, AGENT1_INDEX)

        elif self.agent1_done:
            # agents were met, go to goal state
            if self.meet is True:
                if not self.astar_path_1:
                    # print(self.agent2_pos, self.shared_map[2, 9])
                    self.astar_path_1 = astar(self.fixed_map, (self.agent1_pos[0], self.agent1_pos[1]), self.goal)
                new_pos1 = self.astar_path_1.pop(0)  # Take the next step in the path
                if len(self.astar_path_1) == 0:
                    self._done1 = True

                if new_pos1 is not None:
                    best_action1 = self._get_direction(self.agent1_pos, new_pos1)
                    self._perform_action(best_action1, self.agent1_pos, AGENT1_INDEX)
                    print(self.agent1_pos, new_pos1)
            else:
                # keep looking for agent 2
                best_action1 = self._get_action_towards_goal(self.agent1_pos, AGENT1_INDEX)
                if best_action1 == -1:

                    temp_action = self.action_space.sample()
                    best_action1 = temp_action[1]

                    self._perform_action(best_action1, self.agent1_pos, AGENT1_INDEX)
                    print ("AGENT1 DONE! LOOKING FOR AGENT 2")
                elif best_action1 == -2:
                    self.astar_path_1 = astar(self.fixed_map, (self.agent1_pos[0], self.agent1_pos[1]), self.goal)
                    new_pos1 = self.astar_path_1.pop(0)  # Take the next step in the path

                    if new_pos1 is not None:
                        best_action1 = self._get_direction(self.agent1_pos, new_pos1)
                        self._perform_action(best_action1, self.agent1_pos, AGENT1_INDEX)
                        print(self.agent1_pos, new_pos1)
                else:
                    self._perform_action(best_action1, self.agent1_pos, AGENT1_INDEX)


        if not self.agent2_done:
            if self.meet is True and self.goal is not None:
                if not self.astar_path_2:
                    # print(self.agent2_pos, self.shared_map[2, 9])
                    self.astar_path_2 = astar(self.fixed_map, (self.agent2_pos[0], self.agent2_pos[1]), self.goal)
                new_pos2 = self.astar_path_2.pop(0)  # Take the next step in the path
                if len(self.astar_path_2) == 0:
                    self._done2 = True

                if new_pos2 is not None:
                    best_action2 = self._get_direction(self.agent2_pos, new_pos2)
                    self._perform_action(best_action2, self.agent2_pos, AGENT2_INDEX)
                    print(self.agent2_pos, new_pos2)
            else:
                best_action2 = self._get_action_towards_goal(self.agent2_pos, AGENT2_INDEX)
                if best_action2 == -1:
                    temp_action = self.action_space.sample()
                    best_action2 = temp_action[1]
                # print(self.agent1_done)
                self._perform_action(best_action2, self.agent2_pos, AGENT2_INDEX)

        elif self.agent2_done:
            # agents were met, go to goal state
            if self.meet is True:
                if not self.astar_path_2:
                    # print(self.agent2_pos, self.shared_map[2, 9])
                    self.astar_path_2 = astar(self.fixed_map, (self.agent2_pos[0], self.agent2_pos[1]), self.goal)
                new_pos2 = self.astar_path_2.pop(0)  # Take the next step in the path
                if len(self.astar_path_2) == 0:
                    self._done2 = True

                if new_pos2 is not None:
                    best_action2 = self._get_direction(self.agent2_pos, new_pos2)
                    self._perform_action(best_action2, self.agent2_pos, AGENT2_INDEX)
                    print(self.agent2_pos, new_pos2)
            else:
                # keep looking for agent 1
                best_action2 = self._get_action_towards_goal(self.agent2_pos, AGENT2_INDEX)
                if best_action2 == -1:
                    temp_action = self.action_space.sample()
                    best_action2 = temp_action[1]

                self._perform_action(best_action2, self.agent2_pos, AGENT2_INDEX)


        # Update agent view after moving

        self._update_agent_view(self.agent1_pos, AGENT1_INDEX)

        self._update_agent_view(self.agent2_pos, AGENT2_INDEX)

        if self.fixed_map[self.agent2_pos[0], self.agent2_pos[1]] == GOAL_INDEX:
            self.agent2_done = True

        if self.fixed_map[self.agent1_pos[0], self.agent1_pos[1]] == GOAL_INDEX:
            self.agent1_done = True

        if self._done1 == True and self._done2 == True:
            _done = True

        return self._get_observation(), 0, _done, {}

    def _get_direction(self, current_pos, next_pos):
        # Compute the difference in x and y coordinates
        diff = next_pos - current_pos
        # Find the action that corresponds to this difference and return it
        for action, delta in ACTION_DIRECTIONS.items():
            if np.array_equal(np.array(delta), diff):
                return action

    def _perform_action(self, best_action2, agent_pos, agent_index):
        new_pos2 = self._get_new_position(agent_pos, best_action2, agent_index)
        if self._update_position(agent_pos, new_pos2, agent_index):
            self.shared_map[agent_pos[0], agent_pos[1]] = EMPTY_INDEX
            self.fixed_map[agent_pos[0], agent_pos[1]] = EMPTY_INDEX

            self.shared_map[new_pos2[0], new_pos2[1]] = agent_index
            pos1 = agent_pos[0]
            pos2 = agent_pos[1]

            if self.fixed_map[new_pos2[0], new_pos2[1]] == GOAL_INDEX:

                if agent_index == AGENT2_INDEX:
                    self.agent2_done = True
                elif agent_index == AGENT1_INDEX:
                    self.agent1_done = True

                if not self.meet:
                    self.shared_map[new_pos2[0], new_pos2[1]] = GOAL_INDEX
                    self.fixed_map[new_pos2[0], new_pos2[1]] = GOAL_INDEX

                    self.shared_map[pos1, pos2] = agent_index
                    self.fixed_map[pos1, pos2] = agent_index

                elif self.meet:
                    self.shared_map[agent_pos[0], agent_pos[1]] = EMPTY_INDEX
                    self.fixed_map[agent_pos[0], agent_pos[1]] = EMPTY_INDEX



            else:
                self.fixed_map[new_pos2[0], new_pos2[1]] = agent_index
                self.shared_map[new_pos2[0], new_pos2[1]] = agent_index

                if agent_index == AGENT2_INDEX:
                    self.agent2_pos = new_pos2
                elif agent_index == AGENT1_INDEX:
                    self.agent1_pos = new_pos2

            print(new_pos2)

    def _update_agent_view(self, select_agent_pos, agent_number):
        # Assuming agents can only observe their current position
        if agent_number == 1:

            if self.agent1_done is True:
                if select_agent_pos[0] + 1 < GRID_SIZE:
                    self.shared_map[select_agent_pos[0] + 1, select_agent_pos[1]] = self.fixed_map[
                        select_agent_pos[0] + 1, select_agent_pos[1]]

                if select_agent_pos[0] - 1 >= 0:
                    self.shared_map[select_agent_pos[0] - 1, select_agent_pos[1]] = self.fixed_map[
                        select_agent_pos[0] - 1, select_agent_pos[1]]
                if select_agent_pos[1] + 1 < GRID_SIZE:
                    self.shared_map[select_agent_pos[0], select_agent_pos[1] + 1] = self.fixed_map[
                        select_agent_pos[0], select_agent_pos[1] + 1]

                if select_agent_pos[1] - 1 >= 0:
                    self.shared_map[select_agent_pos[0], select_agent_pos[1] - 1] = self.fixed_map[
                        select_agent_pos[0], select_agent_pos[1] - 1]
                return

            if select_agent_pos[0] + 1 < GRID_SIZE and select_agent_pos[0] + 1 <= self.border:
                self.shared_map[select_agent_pos[0] + 1, select_agent_pos[1]] = self.fixed_map[
                    select_agent_pos[0] + 1, select_agent_pos[1]]

            if select_agent_pos[0] - 1 >= 0:
                self.shared_map[select_agent_pos[0] - 1, select_agent_pos[1]] = self.fixed_map[
                    select_agent_pos[0] - 1, select_agent_pos[1]]

            if select_agent_pos[1] + 1 < GRID_SIZE:
                self.shared_map[select_agent_pos[0], select_agent_pos[1] + 1] = self.fixed_map[
                    select_agent_pos[0], select_agent_pos[1] + 1]

            if select_agent_pos[1] - 1 >= 0:
                self.shared_map[select_agent_pos[0], select_agent_pos[1] - 1] = self.fixed_map[
                    select_agent_pos[0], select_agent_pos[1] - 1]

        if agent_number == 2:

            if self.agent2_done is True:
                if select_agent_pos[0] + 1 < GRID_SIZE:
                    self.shared_map[select_agent_pos[0] + 1, select_agent_pos[1]] = self.fixed_map[
                        select_agent_pos[0] + 1, select_agent_pos[1]]
                if select_agent_pos[0] - 1 >= 0:
                    self.shared_map[select_agent_pos[0] - 1, select_agent_pos[1]] = self.fixed_map[
                        select_agent_pos[0] - 1, select_agent_pos[1]]

                if select_agent_pos[1] + 1 < GRID_SIZE:
                    self.shared_map[select_agent_pos[0], select_agent_pos[1] + 1] = self.fixed_map[
                        select_agent_pos[0], select_agent_pos[1] + 1]
                if select_agent_pos[1] - 1 >= 0:
                    self.shared_map[select_agent_pos[0], select_agent_pos[1] - 1] = self.fixed_map[
                        select_agent_pos[0], select_agent_pos[1] - 1]
                print ("AGENT1 DONE")
                return

            if select_agent_pos[0] + 1 < GRID_SIZE:
                self.shared_map[select_agent_pos[0] + 1, select_agent_pos[1]] = self.fixed_map[
                    select_agent_pos[0] + 1, select_agent_pos[1]]

            if select_agent_pos[0] - 1 >= 0 and select_agent_pos[0] - 1 >= self.border:
                self.shared_map[select_agent_pos[0] - 1, select_agent_pos[1]] = self.fixed_map[
                    select_agent_pos[0] - 1, select_agent_pos[1]]

            if select_agent_pos[1] + 1 < GRID_SIZE:
                self.shared_map[select_agent_pos[0], select_agent_pos[1] + 1] = self.fixed_map[
                    select_agent_pos[0], select_agent_pos[1] + 1]

            if select_agent_pos[1] - 1 >= 0:
                self.shared_map[select_agent_pos[0], select_agent_pos[1] - 1] = self.fixed_map[
                    select_agent_pos[0], select_agent_pos[1] - 1]

    def render(self, mode='human'):
        self.win.fill(WHITE)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                color = WHITE
                if self.fixed_map[i][j] == AGENT1_INDEX and not (self._done1):
                    color = CYAN
                elif self.fixed_map[i][j] == AGENT2_INDEX and not (self._done2):
                    color = BLUE
                elif self.fixed_map[i][j] == GOAL_INDEX:
                    color = GREEN
                elif self.fixed_map[i][j] == OBSTACLE_INDEX:
                    color = RED
                elif self.fixed_map[i][j] == WALL_INDEX:
                    color = BLACK
                pygame.draw.rect(self.win, color,
                                 pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.update()

    def _get_observation(self):
        return self.shared_map

    def _get_new_position(self, pos, action, agent_number):
        new_pos = pos.copy()

        if (self.agent1_done and agent_number == AGENT1_INDEX) or (self.agent2_done and agent_number == AGENT2_INDEX):
            if action == UP and pos[0] > 0:
                new_pos[0] -= 1

            elif action == DOWN and pos[0] < GRID_SIZE - 1:

                new_pos[0] += 1

            elif action == LEFT and pos[1] > 0:
                new_pos[1] -= 1
            elif action == RIGHT and pos[1] < GRID_SIZE - 1:
                new_pos[1] += 1

            return new_pos

        if action == UP and pos[0] > 0:
            if agent_number == AGENT2_INDEX and pos[0] - 1 > self.border:

                new_pos[0] -= 1
            elif agent_number == AGENT1_INDEX:
                new_pos[0] -= 1

        elif action == DOWN and pos[0] < GRID_SIZE - 1:
            if agent_number == AGENT1_INDEX and pos[0] + 1 <= self.border:
                # print (pos[0] , self.border)

                new_pos[0] += 1
            elif agent_number == AGENT2_INDEX:
                new_pos[0] += 1

        elif action == LEFT and pos[1] > 0:
            new_pos[1] -= 1
        elif action == RIGHT and pos[1] < GRID_SIZE - 1:
            new_pos[1] += 1
        return new_pos

    def _update_position(self, pos, new_pos, agent_index):
        if self.fixed_map[new_pos[0], new_pos[1]] == EMPTY_INDEX:
            self.fixed_map[pos[0], pos[1]] = EMPTY_INDEX
            self.fixed_map[new_pos[0], new_pos[1]] = agent_index

            return True
        elif self.fixed_map[new_pos[0], new_pos[1]] == GOAL_INDEX:

            if self.meet:
                self.fixed_map[pos[0], pos[1]] = EMPTY_INDEX
                self.goal = (new_pos[0], new_pos[1])
                self.shared_map[pos[0], pos[1]] = EMPTY_INDEX

            else:
                self.fixed_map[pos[0], pos[1]] = agent_index
                self.shared_map[pos[0], pos[1]] = agent_index
                print("FALSE", agent_index)

            if agent_index == 1:

                self.agent1_done = True
            else:
                self.agent2_done = True
            # self.fixed_map[new_pos[0], new_pos[1]] = agent_index

            return True

        elif self.fixed_map[new_pos[0], new_pos[1]] == WALL_INDEX:
            self.shared_map[new_pos[0], new_pos[1]] = WALL_INDEX
            return False

        elif self.fixed_map[new_pos[0], new_pos[1]] == OBSTACLE_INDEX:
            self.shared_map[new_pos[0], new_pos[1]] = OBSTACLE_INDEX

        return False

    def close(self):
        pygame.quit()


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    closed_list = []

    result = []

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
        closed_list.append(current_node)

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
                0 <= node_position[0] < len(maze),
                0 <= node_position[1] < len(maze[0]),
            ]
            if not all(within_range):
                continue

            if maze[node_position[0]][node_position[1]] == OBSTACLE_INDEX or maze[node_position[0]][node_position[1]] == WALL_INDEX or maze[node_position[0]][node_position[1]] == UNEXPLORED_INDEX:
                continue

            new_node = Node(current_node, node_position)
            children.append(new_node)

        for child in children:
            if child in closed_list:
                continue

            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                    (child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            if len([open_node for open_node in open_list if child == open_node and child.g > open_node.g]) > 0:
                continue

            open_list.append(child)

    # return result


# Main

env = MyCustomEnv()

obs = env.reset()
done = False

while not done:
    actions = env.action_space.sample()
    # print(actions)
    obs, reward, done, info = env.step(actions)

    env.render()
    time.sleep(0.5)
env.close()