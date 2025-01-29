"""
model.py

Holds the neural network architecture for Q-learning.
Kept separate from model.py to maintain a clean, modular codebase.
"""

import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: Dimension of the observation vector
        :param output_dim: Number of discrete actions
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Forward pass. x is a tensor of shape [batch_size, input_dim].
        Returns a tensor of shape [batch_size, output_dim].
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
