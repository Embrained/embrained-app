# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class RSSM(nn.Module):
    def __init__(self, action_dim, obs_dim, hidden_dim=256, state_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # recurrent model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.gru = nn.GRUCell(state_dim + action_dim, hidden_dim)
        
        # prior model: z_t ~ p(z_t | h_t)
        self.prior_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim * 2) # mean and std
        )
        
        # posterior model: z_t ~ q(z_t | h_t, o_t)
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim * 2) # mean and std
        )

    def prior(self, h):
        stats = self.prior_mlp(h)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + 0.1
        return mean, std

    def posterior(self, h, obs):
        inp = torch.cat([h, obs], dim=-1)
        stats = self.post_mlp(inp)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + 0.1
        return mean, std

    def step(self, h_prev, z_prev, a_prev, obs=None):
        # 1. Update belief state
        inp = torch.cat([z_prev, a_prev], dim=-1)
        h_t = self.gru(inp, h_prev)
        
        # 2. Get prior state
        prior_mean, prior_std = self.prior(h_t)
        
        # 3. Get posterior state (if observed)
        if obs is not None:
            post_mean, post_std = self.posterior(h_t, obs)
            # Sample from posterior for training
            dist = D.Normal(post_mean, post_std)
            z_t = dist.rsample()
            return h_t, z_t, (prior_mean, prior_std), (post_mean, post_std)
        else:
            # Sample from prior for imagination
            dist = D.Normal(prior_mean, prior_std)
            z_t = dist.rsample()
            return h_t, z_t, (prior_mean, prior_std), None

class WorldModel(nn.Module):
    def __init__(self, action_dim=2, obs_dim=32, hidden_dim=256, state_dim=32):
        super().__init__()
        self.rssm = RSSM(action_dim, obs_dim, hidden_dim, state_dim)
        
        # Observation predictor: o_t ~ p(o_t | h_t, z_t)
        self.obs_decoder = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # Reward predictor: r_t ~ p(r_t | h_t, z_t)
        self.reward_decoder = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1) # simple MSE loss or categorical
        )

    def forward_state(self, h, z):
        inp = torch.cat([h, z], dim=-1)
        obs_pred = self.obs_decoder(inp)
        rew_pred = self.reward_decoder(inp)
        return obs_pred, rew_pred

class Actor(nn.Module):
    def __init__(self, action_dim=2, hidden_dim=256, state_dim=32):
        super().__init__()
        # Actor uses belief state and latent state
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim * 2) # mean and std for continuous actions
        )
        
    def forward(self, h, z):
        inp = torch.cat([h, z], dim=-1)
        stats = self.net(inp)
        mean, std = torch.chunk(stats, 2, dim=-1)
        # Squash mean to [-1, 1], softplus std
        mean = torch.tanh(mean)
        std = F.softplus(std) + 1e-4
        return mean, std
        
    def sample(self, h, z, deterministic=False):
        mean, std = self.forward(h, z)
        if deterministic:
            return mean
        dist = D.Normal(mean, std)
        return dist.rsample()

class Critic(nn.Module):
    def __init__(self, hidden_dim=256, state_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, h, z):
        inp = torch.cat([h, z], dim=-1)
        return self.net(inp)

class DreamerPolicy(nn.Module):
    def __init__(self, action_dim=2, obs_dim=32, hidden_dim=256, state_dim=32):
        super().__init__()
        self.world_model = WorldModel(action_dim, obs_dim, hidden_dim, state_dim)
        self.actor = Actor(action_dim, hidden_dim, state_dim)
        self.critic = Critic(hidden_dim, state_dim)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
