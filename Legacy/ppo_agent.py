
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared features
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Actor Heads
        self.actor_mu = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim)) 

        # Critic Head
        self.critic = nn.Linear(64, 1)

        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.actor_mu.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        
        mu = self.actor_mu(x)
        log_std = torch.clamp(self.actor_log_std, -20, 2) # Prevent exploding std
        std = log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        
        value = self.critic(x)
        
        return dist, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
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
            action_log_prob = dist.log_prob(action)
            
        return action.detach().numpy()[0], action_log_prob.sum().detach().item(), value.detach().item()

    def update(self, memory):
        # Convert memory to tensors
        old_states = torch.FloatTensor(np.array(memory['states']))
        old_actions = torch.FloatTensor(np.array(memory['actions']))
        old_log_probs = torch.FloatTensor(np.array(memory['log_probs']))
        rewards = memory['rewards']
        is_terminals = memory['is_terminals']
        
        # Monte Carlo estimate of returns
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)
            
        rewards_to_go = torch.FloatTensor(rewards_to_go)
        
        # Normalizing the rewards
        rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            dist, state_values = self.policy(old_states)
            
            action_log_probs = dist.log_prob(old_actions)
            # Match shape if multi-dimensional action space, here it is 1D but sum for joint prob in general
            if action_log_probs.dim() > 1:
                action_log_probs = torch.sum(action_log_probs, dim=1)
                
            dist_entropy = dist.entropy()
            if dist_entropy.dim() > 1:
                dist_entropy = torch.sum(dist_entropy, dim=1)
            
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(action_log_probs - old_log_probs)

            # Finding Surrogate Loss
            advantages = rewards_to_go - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards_to_go) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            self.optimizer.step()
