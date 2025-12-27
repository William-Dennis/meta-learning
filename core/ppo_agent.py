import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor_mu = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.actor_mu.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mu = self.actor_mu(x)
        log_std = torch.clamp(self.actor_log_std, -20, 2)
        std = log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        value = self.critic(x)
        return dist, value


class PPOAgent:
    def __init__(
        self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4
    ):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return (
            action.detach().numpy()[0],
            log_prob.sum().detach().item(),
            value.detach().item(),
        )

    def _compute_returns(self, rewards, is_terminals):
        """Compute discounted returns."""
        returns = []
        discounted = 0
        for reward, terminal in zip(reversed(rewards), reversed(is_terminals)):
            if terminal:
                discounted = 0
            discounted = reward + (self.gamma * discounted)
            returns.insert(0, discounted)
        return torch.FloatTensor(returns)

    def _compute_loss(self, dist, state_values, old_log_probs, old_actions, returns):
        """Compute PPO loss."""
        log_probs = dist.log_prob(old_actions)
        if log_probs.dim() > 1:
            log_probs = torch.sum(log_probs, dim=1)

        entropy = dist.entropy()
        if entropy.dim() > 1:
            entropy = torch.sum(entropy, dim=1)

        state_values = torch.squeeze(state_values)
        ratios = torch.exp(log_probs - old_log_probs)
        advantages = returns - state_values.detach()

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        loss = (
            -torch.min(surr1, surr2)
            + 0.5 * self.mse_loss(state_values, returns)
            - 0.01 * entropy
        )
        return loss

    def update(self, memory):
        """Update policy using PPO."""
        old_states = torch.FloatTensor(np.array(memory["states"]))
        old_actions = torch.FloatTensor(np.array(memory["actions"]))
        old_log_probs = torch.FloatTensor(np.array(memory["log_probs"]))

        returns = self._compute_returns(memory["rewards"], memory["is_terminals"])
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        for _ in range(self.k_epochs):
            dist, state_values = self.policy(old_states)
            loss = self._compute_loss(
                dist, state_values, old_log_probs, old_actions, returns
            )
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
