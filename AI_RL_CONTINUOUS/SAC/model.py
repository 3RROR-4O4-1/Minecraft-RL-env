from torch import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    """
    Outputs mean and log_std for continuous action distribution.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # Clamp the log_std to avoid numerical explosions
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, obs):
        """
        Return a sample action and log_prob under the current policy.
        Uses reparameterization trick.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Sample from Normal(0,1), then scale
        normal = torch.randn_like(mean)
        z = mean + std * normal  # reparameterization
        action = torch.tanh(z)  # optional if we want bounded in [-1,1]

        # log_prob for the tanh distribution
        log_prob = self._log_prob_tanh(normal, mean, std, action)
        return action, log_prob

    def _log_prob_tanh(self, noise, mean, std, action):
        """
        Formula to account for the Tanh squashing in the log-prob.
        You can skip Tanh if you want unbounded actions.
        """
        # log_prob of the Gaussian
        var = std.pow(2)
        log_prob_gaussian = -((noise - 0) ** 2) / (2 * var) - torch.log(std * np.sqrt(2 * np.pi))
        log_prob_gaussian = log_prob_gaussian.sum(dim=1, keepdim=True)

        # Tanh correction
        log_prob = log_prob_gaussian - torch.log(1 - action.pow(2) + 1e-7).sum(dim=1, keepdim=True)
        return log_prob

class Critic(nn.Module):
    """
    Q-value network: Q(s, a).
    We'll create two of these in AI_RL_CONTINUOUS for double Q-learning.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)