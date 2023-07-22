import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import REU_multi_agent_with_RRT
import torch.optim as optim

from envs import MineEnv20x15
import matplotlib.pyplot as plt
import matplotlib
import torch as th
import time
import seaborn as sns
import pandas as pd
import copy
import joblib
import csv

torch.autograd.set_detect_anomaly(True)



class MK_A3C(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=32, gamma=0.99):
        super(MK_A3C, self).__init__()

        self.num_actions = num_actions
        self.gamma = gamma
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, 64)
        self.gru = nn.GRUCell(64, hidden_size)

        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, state, h):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        h = self.gru(x, h)

        action_probs = Categorical(F.softmax(self.actor(h), dim=-1))
        return action_probs, self.critic(h), h

    def act(self, state, h):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs, value, h_updated = self.forward(state, h)
        action = action_probs.sample()
        self.saved_actions.append((action, action_probs.log_prob(action), value))
        return action.item(), h_updated

    def update(self, optimizer):
        R = 0
        returns = []
        policy_losses = []
        value_losses = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for (action, log_prob, value), R in zip(self.saved_actions, returns):
            advantage = R - value.item()

            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward(retain_graph=True)  # Add retain_graph=True here
        optimizer.step()

        del self.rewards[:]
        del self.saved_actions[:]





def train(num_agents, num_episodes, TIMESTEPS, hidden_size, data):
    env = MineEnv20x15(random_target=False)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    episodes_mka3c = []
    success_rates_mka3c = []
    episode_durations_mk = []

    agents = [MK_A3C(num_inputs, num_actions, hidden_size=hidden_size) for _ in range(num_agents)]
    optimizers = [torch.optim.Adam(agent.parameters(), lr=0.0007) for agent in agents]

    h = [torch.zeros(1, agents[i].hidden_size) for i in range(num_agents)]

    for episode in range(num_episodes):
        states = env.reset()
        episodes_mka3c.append(episode + 1)

        done = False
        num_done = 0
        dur = 0


        for t in range (TIMESTEPS):

            actions = []
            h_new = []  # Initialize a new list to hold the updated hidden states


            for i in range(num_agents):

                action, h_updated = agents[i].act(states[i], h[i])
 
                actions.append(action)
                h_new.append(h_updated)
            h = h_new


            next_states, rewards, dones, _ = env.step(actions)

            for i in range(num_agents):
                agents[i].rewards.append(rewards[i])

            if all(dones):
                if dur == 0:
                    dur = t+1

                done = True


            states = next_states

        success_rates_mka3c.append(1 if done else 0)

        if dur == 0:
            dur = TIMESTEPS

        episode_durations_mk.append(dur)

        for i in range(num_agents):
            agents[i].update(optimizers[i])
            h[i] = h[i].detach()


    
    # Here, we calculate the cumulative success rate
    # Calculate the overall success rate
    overall_success_rate = np.sum(success_rates_mka3c) / num_episodes



    # Print overall success rate
    print(f"Overall Success Rate: {overall_success_rate}")

    # Visualize overall success rate
    plt.figure(figsize=(5, 5))
    plt.bar('MK-A3C', overall_success_rate)
    plt.ylim([0, 1])  # This line ensures the y-axis is always scaled from 0 to 1
    plt.ylabel('Overall Success Rate')
    plt.title('Overall Success Rate of MK-A3C')


    plt.savefig('Success_rate_mka3c.png')
    #plt.show()

    # Durations
    # Calculate average durations
    average_duration_mk_a3c = np.mean(episode_durations_mk)

    # Visualize average duration
    plt.figure(figsize=(5, 5))
    plt.bar('MK-A3C', average_duration_mk_a3c)

    plt.ylim([0, 1000])  # This line ensures the y-axis is always scaled from 0 to the max timestep
    plt.ylabel('Average Duration')
    plt.title('Average Duration of MK-A3C')

    plt.savefig('average_duration_mka3c.png')

    data.append(success_rates_mka3c)
    data.append(average_duration_mk_a3c)


data = []
train(4, 3000, 1000, 32, data)
print (data)