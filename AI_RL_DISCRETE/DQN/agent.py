"""
model.py

Defines the 'NeuralSAQAgent' which:
  - Holds a QNetwork (imported from model.py)
  - Implements a simple Q-learning-style update (with replay buffer)
  - Uses epsilon-greedy action selection
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from AI_RL_DISCRETE.DQN.model import QNetwork
from collections import deque



class NeuralDQNAgent:
    def __init__(
            self,
            obs_dim,
            action_size,
            buffer_capacity=10000,   # replay buffer capacity
            batch_size=64,          # how many transitions to sample per update
            learning_rate=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.999
    ):
        self.obs_dim = obs_dim
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-network
        self.q_net = QNetwork(obs_dim, action_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Replay Buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def get_action(self, obs):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(obs_t)  # shape [1, action_size]
            action = torch.argmax(q_values, dim=1).item()
            return action

    def train_step(self, obs, action, reward, next_obs, done):
        """
        Store the transition in replay buffer,
        then sample a batch (if enough data) to do a training step.
        """
        # 1) Store transition
        self.memory.push(obs, action, reward, next_obs, done)

        # 2) If we have enough transitions in the buffer, do a batch update
        if len(self.memory) >= self.batch_size:
            self.train_on_batch()

        # 3) Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_on_batch(self):
        """
        Sample a mini-batch from replay buffer and update Q-network.
        Typical AI_RL_DISCRETE approach using MSE between Q(s,a) and [r + gamma*max_a' Q(s',a')].
        """
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.memory.sample(self.batch_size)

        # Convert to tensors
        obs_t = torch.FloatTensor(obs_batch)         # shape [batch_size, obs_dim]
        actions_t = torch.LongTensor(action_batch)   # shape [batch_size]
        rewards_t = torch.FloatTensor(reward_batch)  # shape [batch_size]
        next_obs_t = torch.FloatTensor(next_obs_batch)
        done_t = torch.FloatTensor(done_batch.astype(np.float32))  # shape [batch_size]

        # Current Q-values
        # shape [batch_size, action_size]
        q_values = self.q_net(obs_t)
        # Q for the chosen actions => shape [batch_size, 1]
        q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Next state Q-values
        with torch.no_grad():
            next_q_values = self.q_net(next_obs_t)  # [batch_size, action_size]
            max_next_q = next_q_values.max(dim=1)[0]  # [batch_size]

        # Target => r + gamma * max(Q(s',a')) * (1 - done)
        target = rewards_t + (1 - done_t) * self.gamma * max_next_q

        # Compute loss
        loss = self.criterion(q_value, target)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, capacity):
        """
        :param capacity: Max number of transitions to store.
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        """
        Add a single transition into the buffer.
        """
        transition = (obs, action, reward, next_obs, done)
        self.memory.append(transition)

    def sample(self, batch_size):
        """
        Sample a batch of transitions randomly.
        Returns five arrays (or lists) for obs, action, reward, next_obs, done.
        """
        batch = random.sample(self.memory, batch_size)

        # Convert to NumPy arrays (or PyTorch tensors) as needed
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = map(np.array, zip(*batch))
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.memory)
