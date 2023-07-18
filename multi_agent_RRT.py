from collections import deque

import gym
from gym import spaces
import numpy as np
import pygame
import time
import random

# Pygame colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
CYAN = (0, 255, 255)
BROWN = (165, 42, 42)

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
PILLAR = 6

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

        self.reward1 = 0.0
        self.reward2 = 0.0

        # Initialize shared map
        self.shared_map = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1],
            [-1, -1, 6, -1, -1, -1, 5, 1, 4, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, 2, -1, -1, -1, -1],
            [-1, -1, 6, -1, -1, -1, -1, -1, -1, -1]
        ], dtype=np.int8)

        # Initialize fixed map layout
        self.fixed_map = np.array([
            [5, 0, 0, 0, 5, 5, 0, 0, 0, 5],
            [5, 5, 5, 0, 0, 5, 5, 5, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 5, 0, 3],
            [0, 0, 5, 5, 5, 5, 0, 0, 0, 5],
            [0, 4, 6, 4, 0, 5, 5, 1, 4, 0],
            [5, 0, 4, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 5, 0, 0, 0, 0, 5, 4, 3],
            [5, 0, 0, 5, 0, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 5],
            [5, 5, 6, 5, 0, 0, 0, 0, 0, 5]
        ], dtype=np.int8)

        self.pillar_loc = [(4, 2), (9, 2)]

        self.border = int((len(self.fixed_map) / 2)) - 1
        self.goal = None
        # print(self.border)

        self.agent_positions = []
        self.goal_list = []
        self.obstacle_positions = []
        self.wall_positions = []

        self.agent1_pos = np.array(np.where(self.fixed_map == AGENT1_INDEX)).reshape(-1)
        self.agent2_pos = np.array(np.where(self.fixed_map == AGENT2_INDEX)).reshape(-1)

        self.obstacle_positions.append((self.agent1_pos[0], self.agent1_pos[1]+1))
        self.obstacle_positions.append((self.agent1_pos[0], self.agent1_pos[1] - 1))

        self.agent1_done = False
        self.agent2_done = False

        self.agent1_inform = False
        self.agent2_inform = False

        self.meet = False
        self.find_agent1 = set()
        self.find_agent2 = set()

        self.agent1_rrt = []
        self.agent2_rrt = []


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

        self.goal_list = []

        self.reward1 = 0.0
        self.reward2 = 0.0

        self.find_agent1 = []
        self.find_agent2 = []

        self.goal_list.append((2, 9))
        self.goal_list.append((6, 9))

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
                        self.reward1 = 1
                        self.reward2 = 1
                        self.meet = True
                        return -2
                    else:
                        return

                # self.meet = False

        return -1  # return a random action if there are no unexplored areas or exits in sight


    def step(self, actions):
        _done = False

        # if agent havent reached the pillars yet
        if not self.agent1_done:

            # print (self.agent1_rrt.plan())
            new_pos1 = (0, 0)
            # if agent meet the obstacles or walls, or RRT havent generate a path
            if len(self.agent1_rrt) == 0:
                # generate 2 path, since we have two pillars
                pillar1_rrt = RRT((self.agent1_pos[0], self.agent1_pos[1]), (self.pillar_loc[0][0], self.pillar_loc[0][1]), self.obstacle_positions)
                pillar2_rrt = RRT((self.agent1_pos[0], self.agent1_pos[1]),
                                  (self.pillar_loc[1][0], self.pillar_loc[1][1]), self.obstacle_positions)


		# since we have two pillars, compare the path and pick the shortest one
                path1 = pillar1_rrt.plan()
                path2 = pillar2_rrt.plan()
                print ("plan1:", path1)

                # compare paths, pick the shorter one 
                if path2 is None:
                    self.agent1_rrt = path1
                elif path1 is None:
                    self.agent1_rrt = path2

                elif len(path1) > len(path2):
                    self.agent1_rrt = path2
                else:
                    self.agent1_rrt = path1
                new_pos1 = self.agent1_rrt.pop(0)


            new_pos1 = self.agent1_rrt.pop(0)  # Take the next step in the path
            if len(self.agent1_rrt) == 0:
                self.agent1_done = True
                self.agent1_rrt = []


            if new_pos1 is not None:
                print ("new_pos1",new_pos1)
                best_action2 = self._get_direction(self.agent1_pos, new_pos1)
                # check if the agent meets the obstacles or not
                can_move = self._perform_action(best_action2, self.agent1_pos, AGENT1_INDEX)
                # the next move is invalid
                if can_move is False:
                    # clear the current path, then generate a new path
                    self.agent1_rrt.clear()
	
	# agent had already reached the pillar, generate a path to the exit
        elif self.agent1_done and not self._done1:
            new_pos1 = (0, 0)
            if not self.astar_path_1:

                path1 = astar(self.shared_map, (self.agent1_pos[0], self.agent1_pos[1]),
                              (self.goal_list[0][0], self.goal_list[0][1]))
                path2 = astar(self.shared_map, (self.agent1_pos[0], self.agent1_pos[1]),
                              (self.goal_list[1][0], self.goal_list[1][1]))

                if path2 is None:
                    self.astar_path_1 = path1
                elif path1 is None:
                    self.astar_path_1 = path2

                elif len(path1) > len(path2):
                    self.astar_path_1 = path2
                else:
                    self.astar_path_1 = path1
                new_pos1 = self.astar_path_1.pop(0)


            new_pos1 = self.astar_path_1.pop(0)  # Take the next step in the path
            # if len(self.astar_path_1) == 0:
            #     self._done1 = True

            if new_pos1 is not None:
                best_action2 = self._get_direction(self.agent1_pos, new_pos1)
                self._perform_action(best_action2, self.agent1_pos, AGENT1_INDEX)


        # if agent havent reached the pillars yet
        if not self.agent2_done:
            new_pos2 = (0, 0)
            # if agent meet the obstacles or walls, or RRT havent generate a path
            if not self.astar_path_2:
                path1 = astar(self.shared_map, (self.agent2_pos[0], self.agent2_pos[1]),
                              (self.pillar_loc[0][0], self.pillar_loc[0][1]))
                path2 = astar(self.shared_map, (self.agent2_pos[0], self.agent2_pos[1]),
                              (self.pillar_loc[1][0], self.pillar_loc[1][1]))

                print ("ASTAR", path2)

                if path2 is None:
                    self.astar_path_2 = path1
                elif path1 is None:
                    self.astar_path_2 = path2

                elif len(path1) > len(path2):
                    self.astar_path_2 = path2
                else:
                    self.astar_path_2 = path1
                new_pos2 = self.astar_path_2.pop(0)

            new_pos2 = self.astar_path_2.pop(0)  # Take the next step in the path
            if len(self.astar_path_2) == 0:
                self.agent2_done = True

            if new_pos2 is not None:
                best_action2 = self._get_direction(self.agent2_pos, new_pos2)

                can_move = self._perform_action(best_action2, self.agent2_pos, AGENT2_INDEX)
                if can_move is False:
                    self.astar_path_2.clear()

        elif self.agent2_done and not self._done2:
            new_pos2 = (0, 0)
            if not self.astar_path_2:

                path1 = astar(self.shared_map, (self.agent2_pos[0], self.agent2_pos[1]),
                              (self.goal_list[0][0], self.goal_list[0][1]))
                path2 = astar(self.shared_map, (self.agent2_pos[0], self.agent2_pos[1]),
                              (self.goal_list[1][0], self.goal_list[1][1]))

                if path2 is None:
                    self.astar_path_2 = path1
                elif path1 is None:
                    self.astar_path_2 = path2

                elif len(path1) > len(path2):
                    self.astar_path_2 = path2
                else:
                    self.astar_path_2 = path1
                new_pos2 = self.astar_path_2.pop(0)

            new_pos2 = self.astar_path_2.pop(0)  # Take the next step in the path
            # if len(self.astar_path_2) == 0:
            #     self._done2 = True

            if new_pos2 is not None:
                best_action2 = self._get_direction(self.agent2_pos, new_pos2)

                can_move = self._perform_action(best_action2, self.agent2_pos, AGENT2_INDEX)
                if can_move is False:
                    self.astar_path_2.clear()


        # Update agent view after moving

        self._update_agent_view(self.agent1_pos, AGENT1_INDEX)

        self._update_agent_view(self.agent2_pos, AGENT2_INDEX)


	# Both reached the exit
        if self._done1 == True and self._done2 == True:
            self.reward1 = 1
            self.reward2 = 1
            _done = True

        return self._get_observation(), (self.reward1, self.reward2), _done, {}

   # determine the action based on the next coordinate
    def _get_direction(self, current_pos, next_pos):
        # Compute the difference in x and y coordinates
        diff = next_pos - current_pos
        print ("diff", current_pos)
        # Find the action that corresponds to this difference and return it
        for action, delta in ACTION_DIRECTIONS.items():
            if np.array_equal(np.array(delta), diff):
                return action

	# update position of agent
    def _update_position(self, pos, new_pos, agent_index):
        if self.fixed_map[new_pos[0], new_pos[1]] == EMPTY_INDEX:

            if agent_index == AGENT1_INDEX:
                self.agent1_pos = new_pos
                self.reward1 = -0.1 / (GRID_SIZE * GRID_SIZE)
            else:
                self.agent2_pos = new_pos
                self.reward2 = -0.1 / (GRID_SIZE * GRID_SIZE)

            self.fixed_map[pos[0], pos[1]] = EMPTY_INDEX
            self.shared_map[pos[0], pos[1]] = EMPTY_INDEX

            self.fixed_map[new_pos[0], new_pos[1]] = agent_index
            self.shared_map[new_pos[0], new_pos[1]] = agent_index
            return True


        elif self.fixed_map[new_pos[0], new_pos[1]] == GOAL_INDEX:
            self.shared_map[new_pos[0], new_pos[1]] = GOAL_INDEX
            # self.meet = True

            if agent_index == AGENT1_INDEX:
                self._done1 = True
                self.reward1 = 1
            else:
                self._done2 = True
                self.reward2 = 1

            self.fixed_map[pos[0], pos[1]] = EMPTY_INDEX
            # self.goal = (new_pos[0], new_pos[1])
            self.shared_map[pos[0], pos[1]] = EMPTY_INDEX
            return True


        elif self.fixed_map[new_pos[0], new_pos[1]] == WALL_INDEX:
            self.shared_map[pos[0], pos[1]] = agent_index
            self.fixed_map[pos[0], pos[1]] = agent_index

            self.shared_map[new_pos[0], new_pos[1]] = WALL_INDEX
            if agent_index == AGENT1_INDEX:
                self.reward1 = -0.1 / (GRID_SIZE * GRID_SIZE)
            else:
                self.reward2 = -0.1 / (GRID_SIZE * GRID_SIZE)
            return False


        elif self.fixed_map[new_pos[0], new_pos[1]] == OBSTACLE_INDEX:
            self.shared_map[pos[0], pos[1]] = agent_index
            self.fixed_map[pos[0], pos[1]] = agent_index

            self.shared_map[new_pos[0], new_pos[1]] = OBSTACLE_INDEX

            if agent_index == AGENT1_INDEX:
                self.reward1 = -0.1 / (GRID_SIZE * GRID_SIZE)
            else:
                self.reward2 = -0.1 / (GRID_SIZE * GRID_SIZE)
            return False

        elif self.fixed_map[new_pos[0], new_pos[1]] == PILLAR:

            self.shared_map[pos[0], pos[1]] = agent_index
            self.shared_map[new_pos[0], new_pos[1]] = PILLAR

            self.fixed_map[pos[0], pos[1]] = agent_index

            if agent_index == AGENT1_INDEX:
                self.agent1_done = True
                self.reward1 = 1
            else:
                self.agent2_done = True
                self.reward2 = 1

            return True


        elif self.fixed_map[new_pos[0], new_pos[1]] == agent_index:

            # print ("AGENT met", (new_pos[0], new_pos[1]), (pos[0], pos[1]))
            if self.fixed_map[new_pos[0], new_pos[1]] == AGENT1_INDEX:
                self.shared_map[new_pos[0], new_pos[1]] = AGENT1_INDEX
            else:
                self.shared_map[new_pos[0], new_pos[1]] == AGENT2_INDEX


            if self.agent2_done and self.agent1_done:
                if len(self.astar_path_2) < len(self.astar_path_1):
                    self.astar_path_1.clear()
                    self.astar_path_1.append((new_pos[0], new_pos[1]))
                    for i in self.astar_path_2:
                        self.astar_path_1.append(i)

                else:
                    self.astar_path_2.clear()
                    for i in self.astar_path_1:
                        self.astar_path_2.append(i)
                return True

            elif self.agent2_done and not self.agent1_done:
                self.agent1_done = True
                self.astar_path_1.clear()
                self.astar_path_1.append((new_pos[0], new_pos[1]))
                for i in self.astar_path_2:
                    self.astar_path_1.append(i)


                return True


            elif self.agent1_done and not self.agent2_done:
                self.agent2_done = True
                self.astar_path_2.clear()
                self.astar_path_2.append((new_pos[0], new_pos[1]))
                for i in self.astar_path_1:
                    self.astar_path_2.append(i)

                return True

            elif not self.agent2_done and not self.agent1_done:
                return False



    def _perform_action(self, best_action2, agent_pos, agent_index):
        new_pos2 = self._get_new_position(agent_pos, best_action2, agent_index)
        # print ("new pos2", new_pos2)
        return self._update_position(agent_pos, new_pos2, agent_index)


	# observe the information at the current state
    def _update_agent_view(self, select_agent_pos, agent_number):
        # Assuming agents can only observe their current position

        if select_agent_pos[0] + 1 < GRID_SIZE:

            self.shared_map[select_agent_pos[0] + 1, select_agent_pos[1]] = self.fixed_map[
                select_agent_pos[0] + 1, select_agent_pos[1]]

            if self.fixed_map[select_agent_pos[0] + 1, select_agent_pos[1]] == WALL_INDEX or self.fixed_map[select_agent_pos[0] + 1, select_agent_pos[1]]\
                    == OBSTACLE_INDEX:
                self.obstacle_positions.append((select_agent_pos[0] + 1, select_agent_pos[1]))

        if select_agent_pos[0] - 1 >= 0:
            self.shared_map[select_agent_pos[0] - 1, select_agent_pos[1]] = self.fixed_map[
                select_agent_pos[0] - 1, select_agent_pos[1]]

            if self.fixed_map[select_agent_pos[0] - 1, select_agent_pos[1]] == WALL_INDEX or self.fixed_map[select_agent_pos[0] - 1, select_agent_pos[1]]\
                    == OBSTACLE_INDEX:
                self.obstacle_positions.append((select_agent_pos[0] - 1, select_agent_pos[1]))

        if select_agent_pos[1] + 1 < GRID_SIZE:
            self.shared_map[select_agent_pos[0], select_agent_pos[1] + 1] = self.fixed_map[
                select_agent_pos[0], select_agent_pos[1] + 1]

            if self.fixed_map[select_agent_pos[0], select_agent_pos[1] + 1] == WALL_INDEX or self.fixed_map[select_agent_pos[0], select_agent_pos[1] + 1]\
                    == OBSTACLE_INDEX:
                self.obstacle_positions.append((select_agent_pos[0], select_agent_pos[1] + 1))

        if select_agent_pos[1] - 1 >= 0:
            self.shared_map[select_agent_pos[0], select_agent_pos[1] - 1] = self.fixed_map[
                select_agent_pos[0], select_agent_pos[1] - 1]

            if self.fixed_map[select_agent_pos[0], select_agent_pos[1] - 1] == WALL_INDEX or self.fixed_map[
                select_agent_pos[0], select_agent_pos[1] - 1] == OBSTACLE_INDEX:

                self.obstacle_positions.append((select_agent_pos[0], select_agent_pos[1] - 1))


	# determine if it is a valid move
    def _get_new_position(self, pos, action, agent_number):
        new_pos = pos.copy()
        # print ("ACTION", action)

        if action == UP and pos[0] > 0:
            new_pos[0] -= 1

        elif action == DOWN and pos[0] < GRID_SIZE - 1:

            new_pos[0] += 1

        elif action == LEFT and pos[1] > 0:
            new_pos[1] -= 1
        elif action == RIGHT and pos[1] < GRID_SIZE - 1:
            new_pos[1] += 1

        return new_pos

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
                elif self.fixed_map[i][j] == PILLAR:
                    color = BROWN
                pygame.draw.rect(self.win, color,
                                 pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.update()

    def _get_observation(self):
        return self.shared_map



    def close(self):
        pygame.quit()




class Node2:
    def __init__(self, position):
        self.position = position
        self.parent = None


class RRT:
    def __init__(self, start, goal, discovered_obstacles, action_space=[(-1, 0), (1, 0), (0, -1), (0, 1)]):
        self.start = Node2(start)  # Create a start node with the given start position
        self.goal = Node2(goal)  # Create a goal node with the given goal position
        self.discovered_obstacles = discovered_obstacles  # List of discovered obstacles
        self.action_space = action_space  # Available actions (up, down, left, right)
        self.nodes = [self.start]  # List to store nodes explored by the RRT

    def plan(self):
        while True:
            random_point = self.get_random_point()  # Get a random point in the maze
            nearest_node = self.get_nearest_node(random_point)  # Find the nearest node to the random point
            new_node = self.steer(nearest_node, random_point)  # Steer towards the random point from the nearest node

            if self.collision_free(nearest_node, new_node):  # Check if the movement from nearest_node to new_node is collision-free
                self.nodes.append(new_node)  # Add the new node to the list of explored nodes
                if self.reached_goal(new_node):  # Check if the new node is close enough to the goal
                    break

        path = self.extract_path(self.nodes[-1])  # Extract the path by following parent pointers starting from the last node in nodes
        return path

    def get_random_point(self):
        # We assume the maze size is 10x10
        return (random.randint(0, 10), random.randint(0, 10))  # Generate a random point within the maze bounds

    def get_nearest_node(self, random_point):
        return min(self.nodes, key=lambda node: self.distance(node.position, random_point))  # Find the node with the minimum distance to the random point

    def steer(self, nearest_node, random_point):
        best_next_position = nearest_node.position
        min_dist = self.distance(nearest_node.position, random_point)

        for action in self.action_space:  # Try all available actions (movement directions)
            next_position = tuple(map(sum, zip(nearest_node.position, action)))  # Calculate the next position by adding the action to the nearest node's position
            if self.distance(next_position, random_point) < min_dist:  # Check if the distance between the next position and the random point is smaller than the minimum distance
                min_dist = self.distance(next_position, random_point)
                best_next_position = next_position

        new_node = Node2(best_next_position)  # Create a new node with the best next position as its position
        new_node.parent = nearest_node  # Set the nearest node as the parent of the new node
        return new_node

    def collision_free(self, nearest_node, new_node):
        # If the movement from nearest_node to new_node does not collide with the pillars
        return new_node.position not in self.discovered_obstacles and new_node.position[0] < GRID_SIZE and new_node.position[1] < GRID_SIZE

    def reached_goal(self, new_node):
        # Check if the new node is close enough to the goal
        return self.distance(new_node.position, self.goal.position) < 1  # Assuming a distance of 1 is close enough

    def extract_path(self, node):
        path = []
        while node is not None:
            path.append(node.position)  # Add the position of the node to the path
            node = node.parent  # Move to the parent node

        return path[::-1]  # Reverse the path to get the correct order

    @staticmethod
    def distance(position_a, position_b):
        return ((position_a[0] - position_b[0]) ** 2 + (position_a[1] - position_b[1]) ** 2) ** 0.5  # Calculate the Euclidean distance between two positions





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
    start_node = Node(None, start)  # Create a start node with no parent and the given start position
    end_node = Node(None, end)  # Create an end node with no parent and the given end position

    open_list = []  # List to store nodes that are open (not yet visited)
    closed_list = []  # List to store nodes that are closed (visited)

    result = []  # List to store the result path

    open_list.append(start_node)  # Add the start node to the open list

    while len(open_list) > 0:  # Continue until there are nodes to visit in the open list
        current_node = open_list[0]  # Get the first node from the open list
        result.append(current_node.position)  # Add the current node's position to the result list

        current_index = 0
        for index, item in enumerate(open_list):  # Find the node with the lowest f value in the open list
            if item.f < current_node.f:
                current_node = item
                current_index = index

        open_list.pop(current_index)  # Remove the current node from the open list
        closed_list.append(current_node)  # Add the current node to the closed list

        if current_node == end_node:  # If the current node is the end node, construct the path and return it
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Reverse the path before returning

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Check adjacent positions (up, down, left, right)
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            within_range = [
                0 <= node_position[0] < len(maze),  # Check if the new position is within the maze boundaries
                0 <= node_position[1] < len(maze[0]),
            ]
            if not all(within_range):
                continue

            if maze[node_position[0]][node_position[1]] == OBSTACLE_INDEX or maze[node_position[0]][node_position[1]] == WALL_INDEX:
                continue  # Skip this position if it is an obstacle or a wall

            new_node = Node(current_node, node_position)  # Create a new node with the current node as the parent
            children.append(new_node)  # Add the new node to the list of children

        for child in children:
            if child in closed_list:
                continue  # Skip this child if it has already been visited

            child.g = current_node.g + 1  # Set the g value of the child (cost from start to current node + 1)
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                    (child.position[1] - end_node.position[1]) ** 2)  # Calculate the h value (heuristic distance to end node)
            child.f = child.g + child.h  # Calculate the f value (g + h)

            if len([open_node for open_node in open_list if child == open_node and child.g > open_node.g]) > 0:
                continue  # Skip this child if there is already a better path to it in the open list

            open_list.append(child)  # Add the child to the open list



# Main

env = MyCustomEnv()

obs = env.reset()
done = False

env.render()
while not done:
    actions = env.action_space.sample()
    # print(actions)
    obs, reward, done, info = env.step(actions)
    print (reward)


    time.sleep(0.5)
    env.render()
env.close()
