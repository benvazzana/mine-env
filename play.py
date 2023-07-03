from envs import MineEnv20x15
import pygame

from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':
    env = MineEnv20x15(random_target=False)
    check_env(env)

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
            if keys[pygame.K_a]:
                action = [0.5]
            if keys[pygame.K_d]:
                action = [-0.5]

            observation, reward, done, info = env.step(action)
            #if reward > 0:
                #env.print_obs(observation)
            score += reward
            steps += 1
        print('Episode: {}, Score: {}, Steps: {}'.format(episode, score, steps))