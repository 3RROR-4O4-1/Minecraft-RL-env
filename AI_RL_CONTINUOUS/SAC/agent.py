import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from AI_RL_CONTINUOUS.SAC.model import Actor, Critic


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = map(np.array, zip(*batch))
        return s, a, r, s_next, d

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(
            self,
            obs_dim,
            action_dim,
            lr=1e-3,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            buffer_capacity=100000,
            batch_size=64
    ):
        """
        :param obs_dim: dimension of observations
        :param action_dim: dimension of continuous actions
        :param lr: learning rate
        :param gamma: discount factor
        :param tau: soft-update factor for target networks
        :param alpha: entropy regularization coefficient (if not auto-tuned)
        :param buffer_capacity: capacity of replay buffer
        :param batch_size: training batch size
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.lr = lr
        self.buffer_capacity = buffer_capacity

        # Actor and Critics
        self.actor = Actor(obs_dim, action_dim)
        self.critic1 = Critic(obs_dim, action_dim)
        self.critic2 = Critic(obs_dim, action_dim)

        # Target critics
        self.critic1_target = Critic(obs_dim, action_dim)
        self.critic2_target = Critic(obs_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, obs, eval_mode=False):
        """
        :param obs: np.array shape [obs_dim]
        :param eval_mode: if True, use mean action (no noise)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        if eval_mode:
            with torch.no_grad():
                mean, log_std = self.actor(obs_t)
                # deterministic = mean
                action = torch.tanh(mean)  # shape [1, action_dim]
            return action.squeeze(0).cpu().numpy()
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(obs_t)
            return action.squeeze(0).cpu().numpy()

    def update(self):
        """
        One gradient update of the SAC (actor, 2 critics).
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        s, a, r, s_next, d = self.replay_buffer.sample(self.batch_size)
        s_t = torch.FloatTensor(s)
        a_t = torch.FloatTensor(a)
        r_t = torch.FloatTensor(r).unsqueeze(-1)
        s_next_t = torch.FloatTensor(s_next)
        d_t = torch.FloatTensor(d).unsqueeze(-1)

        # === Critic Loss ===
        with torch.no_grad():
            # Next action & next log-prob
            a_next, log_prob_next = self.actor.sample(s_next_t)
            # Target Q-values
            q_next_1 = self.critic1_target(s_next_t, a_next)
            q_next_2 = self.critic2_target(s_next_t, a_next)
            q_next = torch.min(q_next_1, q_next_2) - self.alpha * log_prob_next
            # Bellman backup
            target_q = r_t + (1 - d_t) * self.gamma * q_next

        # Current Q estimates
        q1 = self.critic1(s_t, a_t)
        q2 = self.critic2(s_t, a_t)
        critic1_loss = nn.MSELoss()(q1, target_q)
        critic2_loss = nn.MSELoss()(q2, target_q)

        # Optimize critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # === Actor Loss ===
        a_curr, log_prob = self.actor.sample(s_t)
        q1_curr = self.critic1(s_t, a_curr)
        q2_curr = self.critic2(s_t, a_curr)
        q_curr = torch.min(q1_curr, q2_curr)
        actor_loss = (self.alpha * log_prob - q_curr).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # === Soft Update Targets ===
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def store_transition(self, s, a, r, s_next, done):
        self.replay_buffer.push(s, a, r, s_next, done)

    # -------------------------------------------------------------------------
    # Saving & Loading the Agent
    # -------------------------------------------------------------------------
    def save(self, filepath="sac_agent.pth"):
        """
        Save the entire agent state (networks + optimizers + hyperparams).
        If you also want to store your replay buffer, you can do that here.
        """
        checkpoint = {
            # Hyperparameters
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "tau": self.tau,
            "alpha": self.alpha,
            "buffer_capacity": self.buffer_capacity,
            "batch_size": self.batch_size,

            # Networks
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),

            # Optimizers
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),

            # If you want to store the buffer, be careful with large memory usage:
            # "replay_buffer": list(self.replay_buffer.buffer),
        }

        torch.save(checkpoint, filepath)
        print(f"SACAgent saved to {filepath}")

    @classmethod
    def load_from_checkpoint(cls, filepath):
        """
        Create a new SACAgent from a saved checkpoint.
        This reinitializes the networks & optimizer from the file.
        """
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

        # Create a new agent with the same hyperparameters
        agent = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            lr=checkpoint["lr"],
            gamma=checkpoint["gamma"],
            tau=checkpoint["tau"],
            alpha=checkpoint["alpha"],
            buffer_capacity=checkpoint["buffer_capacity"],
            batch_size=checkpoint["batch_size"]
        )

        # Load network weights
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        agent.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        agent.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        agent.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])

        # Load optimizer states
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        agent.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
        agent.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state_dict"])

        print(f"SACAgent loaded from {filepath}")
        return agent
