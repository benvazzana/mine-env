from envs import MineEnv20x15
import pygame

if __name__ == '__main__':
    env = MineEnv20x15(random_target=True)

    episodes = 10
    for episode in range(1, episodes+1):
        steps = 0
        observation = env.reset()
        reward = 0
        score = 0
        action = 0
        done = False
        while not done:
            env.render()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] and keys[pygame.K_a]:
                action = 315
            elif keys[pygame.K_w] and keys[pygame.K_d]:
                action = 45
            elif keys[pygame.K_s] and keys[pygame.K_a]:
                action = 225
            elif keys[pygame.K_s] and keys[pygame.K_d]:
                action = 135
            elif keys[pygame.K_w]:
                action = 0
            elif keys[pygame.K_a]:
                action = 270
            elif keys[pygame.K_s]:
                action = 180
            elif keys[pygame.K_d]:
                action = 90

            observation, reward, done, info = env.step(action)
            score += reward
            steps += 1
        print('Episode: {}, Score: {}, Steps: {}'.format(episode, score, steps))