from envs import MineEnv20x15
import pygame

from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':
    env = MineEnv20x15()
    # Default exit location: (19, 14)
    env.random_targets = False
    # Default agent location: (0, 0)
    env.randomize_agent = True
    env.mine_layout.cell_shifts = 30

    episodes = 10
    for episode in range(1, episodes+1):
        steps = 0
        observation = env.reset()
        reward = 0
        score = 0
        done = False
        while not done:
            action = [0]
            env.render()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                observation, reward, done, info = env.step(0)
                #env.print_obs(observation)
            if keys[pygame.K_a]:
                observation, reward, done, info = env.step(1)
                #env.print_obs(observation)
            if keys[pygame.K_s]:
                observation, reward, done, info = env.step(2)
                #env.print_obs(observation)
            if keys[pygame.K_d]:
                observation, reward, done, info = env.step(3)
                #env.print_obs(observation)
            #if reward > 0:
            score += reward
            steps += 1
        print('Episode: {}, Score: {}, Steps: {}'.format(episode, score, steps))
