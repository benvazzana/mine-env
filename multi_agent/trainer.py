import numpy as np

import gym
from gym.vector import AsyncVectorEnv
from gym.wrappers import TimeLimit


from envs import MineEnv20x15
from MADDPG import MADDPG
from pytorch_DRL.MAA2C import MAA2C
from pytorch_DRL.common.utils import agg_double_list
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
import torch as th
import time
import seaborn as sns
import pandas as pd
import csv

model = None
#matplotlib.use('Agg')



# Reference
# The Algorithms I'm using for training is collected from:
# https://github.com/ChenglongChen/pytorch-DRL/blob/master/MAA2C.py, https://github.com/xuehy/pytorch-maddpg/blob/master/MADDPG.py


if __name__ == '__main__':
    env = MineEnv20x15(random_target=False)
    # roll out n steps
    ROLL_OUT_N_STEPS = 1000
    MEMORY_CAPACITY = ROLL_OUT_N_STEPS

    MAX_EPISODES = 3000
    EPISODES_BEFORE_TRAIN = 10
    EVAL_EPISODES = 100
    EVAL_INTERVAL = 100


    mam2c_model = MAA2C(env, 4, env.observation_space.shape[0], env.action_space.n, roll_out_n_steps=ROLL_OUT_N_STEPS, use_cuda = True, training_strategy="centralized", actor_parameter_sharing=True, critic_parameter_sharing=True)
    episodes = []
    eval_rewards = []

    # Lists to keep track of metrics for MAA2C
    episodes_maa2c = []
    success_rates_maa2c = []
    episode_durations_maa2c = []



    while mam2c_model.n_episodes < MAX_EPISODES:
        print ("Episode: ", mam2c_model.n_episodes)
        #start_time = time.time()

        mam2c_model.interact()

        #episode_duration = time.time() - start_time
        if mam2c_model.episode_done:
            print ("DONE!")
        success_rates_maa2c.append(1 if mam2c_model.episode_done else 0)  # success should be 1 for success and 0 for failure
        episode_durations_maa2c.append(mam2c_model.timestep)
        episodes_maa2c.append(mam2c_model.n_episodes + 1)

        if mam2c_model.n_episodes >= EPISODES_BEFORE_TRAIN:
            mam2c_model.train()



    episodes_maddpg = []
    success_rates_maddpg = []
    episode_durations_maddpg = []
    #th.save(mam2c_model, '/home/aghnw/anaconda3/envs/myWork11//maa2c_model.pth')

    env = MineEnv20x15(random_target=False)
    maddpg = MADDPG(4, env.observation_space.shape[0], env.action_space.n, 32, 10000,
                    10)
    reward_record = []
    
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

    for i_episode in range(MAX_EPISODES):
        episodes_maddpg.append(i_episode + 1)
        obs = env.reset()
        obs = np.stack(obs)

        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()

        total_reward = 0.0
        rr = np.zeros((4,))

        t = 0
        check_done = False
        for t in range(ROLL_OUT_N_STEPS):
          

            if obs is None:
                break

            obs = obs.type(FloatTensor)
           
            action = maddpg.select_action(obs).data.cpu()


            
            obs_, reward, done, _ = env.step(action.numpy())

            reward = th.FloatTensor(reward).type(FloatTensor)
            obs_ = np.stack(obs_)
            obs_ = th.from_numpy(obs_).float()

            # if it has next state
            if t != ROLL_OUT_N_STEPS - 1:
                next_obs = obs_
            else:
                next_obs = None

            total_reward += reward.sum()
            rr += reward.cpu().numpy()
            actions = th.eye(env.action_space.n)[action]

            maddpg.memory.push(obs.data, actions, next_obs, reward)
            obs = next_obs

            c_loss, a_loss = maddpg.update_policy()

            if all(done):
                check_done = True

                break
        success_rates_maddpg.append(1 if check_done is True else 0)
        episode_durations_maddpg.append(t+1)
        maddpg.episode_done += 1
        print('Episode: %d, reward = %f' % (i_episode, total_reward))
        reward_record.append(total_reward)


    print ("episode_durations_maddpg", episode_durations_maddpg)
    print ("episode_durations_maa2c", episode_durations_maa2c)
    print("success_rates_maa2c", success_rates_maa2c)
    print("success_rates_maddpg", success_rates_maddpg)

    # Success rates
    # Compute overall success rates for both algorithms
    overall_success_rate_maa2c = np.sum(success_rates_maa2c) / MAX_EPISODES
    overall_success_rate_maddpg = np.sum(success_rates_maddpg) / MAX_EPISODES

    # Set up bar names and heights
    bar_names = ['MA-A2C', 'MADDPG']
    bar_heights = [overall_success_rate_maa2c, overall_success_rate_maddpg]

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(5, 5))

    # Create the bar chart
    ax.bar(bar_names, bar_heights, color=['blue', 'orange'])

    # Customize the chart
    ax.set_ylim([0, 1])  # This line ensures the y-axis is always scaled from 0 to 1
    ax.set_ylabel('Overall Success Rate')
    ax.set_title('Overall Success Rate of MA-A2C and MADDPG')

    # Save the chart to a file
    fig.savefig('success_rate.png')

    

    # Durations
    # Calculate average durations
    average_duration_maa2c = np.mean(episode_durations_maa2c)
    average_duration_maddpg = np.mean(episode_durations_maddpg)

    # Set up bar names and heights for duration
    duration_bar_names = ['MA-A2C', 'MADDPG']
    duration_bar_heights = [average_duration_maa2c, average_duration_maddpg]

    # Create the figure and axis objects for duration
    duration_fig, duration_ax = plt.subplots(figsize=(5, 5))

    # Create the bar chart for duration
    duration_ax.bar(duration_bar_names, duration_bar_heights, color=['blue', 'orange'])

    # Customize the duration chart
    duration_ax.set_ylabel('Average Episode Duration')
    duration_ax.set_title('Average Episode Duration of MA-A2C and MADDPG')

    # Save the duration chart to a file
    duration_fig.savefig('average_duration.png')
    #plt.show()

