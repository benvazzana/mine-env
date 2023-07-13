from mine import MineLayout, MineEnv
from envs import MineEnv20x15
import pygame
import trainer


if __name__ == '__main__':
    env = MineEnv20x15(random_target=False)
    model = trainer.make_a2c_model('a2c-callback2/best_model', env=env, n_envs=1)

    episodes = 10
    for episode in range(1, episodes+1):
        observation = env.reset()
        reward = 0
        score = 0
        done = False
        action = 0
        while not done:
            env.render()
            keys = pygame.key.get_pressed()
            action, _states = model.predict(observation)
            observation, reward, done, info = env.step(int(action))
            score += reward
        print('Episode: {}, Score: {}'.format(episode, score))