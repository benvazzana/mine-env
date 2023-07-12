import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import torch.optim as optim


class MK_A3C(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(MK_A3C, self).__init__()

        self.num_actions = num_actions

        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, 64)
        self.gru = nn.GRUCell(64, 32)

        self.actor = nn.Linear(32, num_actions)
        self.critic = nn.Linear(32, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x, h):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        h = self.gru(x, h)

        action_prob = torch.distributions.Normal(self.actor(h), torch.exp(self.critic(h)))
        return action_prob, h

    def act(self, state, h):
        state = torch.from_numpy(state).float()
        action_prob, h = self.forward(state, h)
        action = action_prob.sample()
        self.saved_actions.append(action_prob.log_prob(action))
        return action.item(), h

    def update(self, optimizer):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # List to save actor (policy) loss
        value_losses = []  # List to save critic (value) loss
        returns = []  # List to save the true values

        # Calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Calculate actor and critic losses and perform backpropagation
        for log_prob, value, R in zip(saved_actions, self.values, returns):
            advantage = R - value.item()

            # Calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # Calculate critic (value) loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # Reset gradients
        optimizer.zero_grad()

        # Sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # Perform backprop
        loss.backward()
        optimizer.step()

        # Reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
