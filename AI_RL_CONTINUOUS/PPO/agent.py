import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple

from AI_RL_CONTINUOUS.PPO.model import ActorCritic


# A small structure to hold rollout data
Transition = namedtuple(
    'Transition',
    ['state', 'action', 'log_prob', 'reward', 'next_state', 'done', 'value']
)

class RolloutBuffer:
    """
    Stores transitions from the environment until it's time to update.
    We then compute advantages and returns for PPO.
    """
    def __init__(self):
        self.transitions = []

    def add(self, transition):
        self.transitions.append(transition)

    def clear(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        eps_clip=0.2,
        K_epochs=10,
        batch_size=64,
        hidden_dim=128
    ):
        """
        :param obs_dim: State/observation dimension
        :param action_dim: Action dimension
        :param lr: learning rate
        :param gamma: discount factor
        :param lam: GAE(lambda) factor
        :param eps_clip: PPO clip param (epsilon)
        :param K_epochs: number of epochs to update over each batch
        :param batch_size: minibatch size
        :param hidden_dim: hidden size for the network
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # PPO hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor-Critic network
        self.ac = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

    def select_action(self, obs, eval_mode=False):
        """
        :param obs: np.array, shape [obs_dim]
        :param eval_mode: if True, use mean (deterministic) action
        :return: action in [-1,1], log_prob, value
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, log_std, value = self.ac.forward(obs_t)
            std = log_std.exp()
            if eval_mode:
                # Deterministic = mean. Then clamp with tanh
                raw_action = mean
                action = torch.tanh(raw_action)
                # log_prob for deterministic is optional (we often skip it in eval mode)
                log_prob = None
            else:
                normal_dist = torch.distributions.Normal(mean, std)
                raw_action = normal_dist.rsample()
                action = torch.tanh(raw_action)

                log_prob = normal_dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
                log_prob -= torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1, keepdim=True)

        return (
            action.cpu().numpy().flatten(),
            log_prob.item() if log_prob is not None else 0.0,
            value.item(),
        )

    def store_transition(self, transition):
        """
        transition: a namedtuple('Transition', [state, action, log_prob, reward, next_state, done, value])
        """
        self.buffer.add(transition)

    def finish_trajectory(self):
        """
        Once an episode or a batch ends, compute advantage (GAE) and returns.
        We'll store them in the buffer as extra fields advantage & return_.
        """
        transitions = self.buffer.transitions
        # Convert to arrays for easy processing
        rewards = [t.reward for t in transitions]
        values = [t.value for t in transitions]
        dones = [t.done for t in transitions]

        # We also need the "next value" for the last step
        # In on-policy PPO, you can handle the last step's next_value as 0 if done,
        # or do a "bootstrap" if not done. Let's assume done for each trajectory
        # or handle partial episodes. For a more general approach,
        # pass in next_value from outside if not done.
        advantages = []
        gae = 0
        for i in reversed(range(len(transitions))):
            mask = 1.0 - float(dones[i])
            # next_value is 0 if done else values[i+1], but watch out for last index
            if i == len(transitions) - 1:
                next_value = 0.0
            else:
                next_value = values[i+1]

            delta = rewards[i] + self.gamma * next_value * mask - values[i]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.append(gae)
        advantages.reverse()

        # Store advantage & return in each transition
        for i in range(len(transitions)):
            returns_i = values[i] + advantages[i]
            self.buffer.transitions[i] = transitions[i]._replace(
                advantage=advantages[i],
                return_=returns_i
            )

    def update(self):
        """
        Perform the PPO update (clipped surrogate objective).
        - We'll do multiple epochs (self.K_epochs) over the data
        - Each epoch, we shuffle and create minibatches
        """
        # Convert transitions to torch tensors
        states = []
        actions = []
        log_probs_old = []
        advantages = []
        returns = []
        for t in self.buffer.transitions:
            states.append(t.state)
            actions.append(t.action)
            log_probs_old.append(t.log_prob)
            advantages.append(t.advantage)
            returns.append(t.return_)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).unsqueeze(-1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)

        # Normalize advantages (common trick)
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Training in mini-batches for self.K_epochs
        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_log_probs_old = log_probs_old[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Evaluate new log_probs, entropy, values
                log_probs, dist_entropy, values = self.ac.evaluate_actions(
                    batch_states, batch_actions
                )
                # PPO ratio
                ratio = torch.exp(log_probs - batch_log_probs_old)

                # Clip objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (MSE)
                critic_loss = nn.MSELoss()(values, batch_returns)

                # Entropy bonus (optional)
                entropy_bonus = -0.0 * dist_entropy  # set weight if desired

                loss = actor_loss + 0.5 * critic_loss + entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear buffer after update
        self.buffer.clear()

    def save(self, filepath="./storage/ppo_agent.pth"):
        checkpoint = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "gamma": self.gamma,
            "lam": self.lam,
            "eps_clip": self.eps_clip,
            "K_epochs": self.K_epochs,
            "batch_size": self.batch_size,
            "lr": self.optimizer.defaults["lr"],

            "actor_critic_state_dict": self.ac.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")

    @classmethod
    def load_from_checkpoint(cls, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        agent = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            lr=checkpoint["lr"],
            gamma=checkpoint["gamma"],
            lam=checkpoint["lam"],
            eps_clip=checkpoint["eps_clip"],
            K_epochs=checkpoint["K_epochs"],
            batch_size=checkpoint["batch_size"],
            hidden_dim=checkpoint["hidden_dim"],
        )
        agent.ac.load_state_dict(checkpoint["actor_critic_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Agent reinitialized and loaded from {filepath}")
        return agent




