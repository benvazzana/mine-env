import numpy as np
import os
import signal
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

model = None

policy_kwargs = dict(net_arch=[dict(pi=[128, 128, 128],
                                    vf=[128, 128, 128])])

def sig_handler(sig, frame):
    save = input('Save model? [y/n] ')[0] == 'y'
    if save and model is not None:
        path = 'models/{}'.format(input('Model name: '))
        model.save(path)
    sys.exit()

def make_env():
    return TimeLimit(MineEnv20x15(random_target=False), max_episode_steps=1000)

def make_a2c_model(name=None, env=None, n_envs=8):
    if env is None:
        env = make_vec_env(lambda: make_env(), n_envs=n_envs)
    if name is None:
        return A2C('MlpPolicy', env, n_steps=8, gae_lambda=0.9, vf_coef=0.4,
                   policy_kwargs=policy_kwargs, verbose=1, tensorboard_log='./tensorboard-logs/a2c/')
    return A2C.load('models/{}'.format(name), env, n_envs=n_envs)

def make_ddpg_model(name=None, env=None, n_envs=8):
    if env is None:
        env = make_vec_env(lambda: make_env(), n_envs=n_envs)
    if name is None:
        return DDPG('MlpPolicy', env, gamma=0.98, buffer_size=200000, action_noise=NormalActionNoise(mean=0.25, sigma=0.1), learning_starts=20000, policy_kwargs=dict(net_arch=[400, 300]), verbose=1, tensorboard_log='./tensorboard-logs/ddpg/')
    return DDPG.load('models/{}'.format(name), env, n_envs=n_envs)

def make_dqn_model(name=None, env=None, n_envs=8):
    if env is None:
        env = make_vec_env(lambda: make_env(), n_envs=n_envs)
    if name is None:
        return DQN('MlpPolicy', env, batch_size=128, buffer_size=50000, learning_starts=100000, exploration_fraction=0.12, exploration_final_eps=0.1, verbose=1, tensorboard_log='./tensorboard-logs/dqn/')
    return DQN.load('models/{}'.format(name), env, n_envs=n_envs)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sig_handler)

    vec_env = make_vec_env(lambda: make_env(), n_envs=os.cpu_count())
    model = make_a2c_model(env=vec_env, n_envs=os.cpu_count())

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
    eval_callback = EvalCallback(vec_env,
                                 callback_on_new_best=stop_callback,
                                 eval_freq=10000,
                                 best_model_save_path='models/a2c-128x3',
                                 verbose=1)

    model.learn(total_timesteps=5000000, callback=eval_callback)