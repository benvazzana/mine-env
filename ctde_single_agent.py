import sys

import gym
from gym.vector import AsyncVectorEnv
from gym.wrappers import TimeLimit

from stable_baselines3 import A2C, DQN, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.noise import NormalActionNoise

from envs import MineEnv20x15

env = MineEnv20x15(random_target=False)
model = A2C('MlpPolicy', env, verbose=1)

# Train the model with increasing difficulty
for i_episode in range(3000):
    obs = env.reset()
    total_rewards = 0
    for t in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        model.learn(total_timesteps=1000)
        total_rewards += rewards
        if dones:
            print("Episode finished after {} timesteps".format(t+1))
            break

model.save('models/a2c-rrtx')

# Load the trained model
model = A2C.load('models/a2c-rrtx')

# Create the environment
env = MineEnv20x15(random_target=False)

# Number of episodes to play
n_episodes = 100

for i_episode in range(n_episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Predict the action with the model
        action, _states = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, done, info = env.step(action)

        total_reward += reward

    print(f"Episode {i_episode + 1}: Total Reward = {total_reward}")

env.close()