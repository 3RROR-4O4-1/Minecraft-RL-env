import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """
    A combined Actor + Critic network for PPO in continuous action spaces.
    - Actor outputs mean, log_std for each action dim. (We apply tanh to produce [-1,1].)
    - Critic outputs a scalar value function V(s).
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # Common feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))  # trainable log_std

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        """
        Returns:
          - means of action distribution (shape [batch, action_dim])
          - log_std (shape [action_dim], broadcast over batch)
          - value estimate (shape [batch, 1])
        """
        features = self.shared(obs)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std  # shape [action_dim]
        value = self.critic(features)
        return mean, log_std, value

    def get_action_log_prob_value(self, obs):
        """
        Given obs, sample an action (with reparameterization or direct sampling),
        return:
          action, log_prob(action), value(s).
        """
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()

        # Sample from Normal distribution
        normal_dist = torch.distributions.Normal(mean, std)
        raw_action = normal_dist.rsample()  # reparam trick
        # Tanh to bound in [-1, 1]
        action = torch.tanh(raw_action)

        # Log prob:
        #   log_prob(raw_action) - sum(log(1 - a^2)) for tanh transform
        log_prob = normal_dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        # Tanh correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        """
        Used for PPO update:
          - obs: shape [batch, obs_dim]
          - actions: shape [batch, action_dim] in [-1, 1]
        Returns:
          - log_probs of those actions
          - entropy
          - state value
        """
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()

        # We'll invert the tanh via atanh: raw_action = arctanh(actions)
        # Since actions in [-1,1], we do atanh. Let's clamp to avoid numerical issues.
        eps = 1e-6
        atanh_actions = 0.5 * torch.log((1 + actions).clamp(min=eps) / (1 - actions).clamp(min=eps))

        normal_dist = torch.distributions.Normal(mean, std)
        log_probs = normal_dist.log_prob(atanh_actions).sum(dim=-1, keepdim=True)
        # Tanh correction
        log_probs -= torch.log(1 - actions.pow(2) + 1e-7).sum(dim=-1, keepdim=True)

        # For entropy, we can approximate the Normal's entropy,
        # but strictly we should also factor in the tanh transform.
        # A simpler approach is to measure the -log_prob(action) average.
        # For PPO clipping, this is typically enough. We'll do a simpler measure:
        dist_entropy = normal_dist.entropy().sum(dim=-1).mean()

        return log_probs, dist_entropy, value