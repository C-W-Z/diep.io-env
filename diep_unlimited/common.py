# Common imports
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import itertools
import time
from collections import deque
from math import sqrt


# Networks
LOG_STD_MIN = -20
LOG_STD_MAX = 2

HIDDEN_LAYER = 256

# Xavier init
def weights_init_(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain = 1)
        nn.init.constant_(layer.bias, 0)


class FCFF(nn.Module):
    def __init__(self, dim_in, dim_out, hidden = HIDDEN_LAYER, n_hidden = 1):
        super().__init__()

        layers = [nn.Linear(dim_in, hidden), nn.ReLU()]
        for _ in range(n_hidden):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]

        layers += [nn.Linear(hidden, dim_out)]
        self.model = nn.Sequential(*layers)
        self.apply(weights_init_)


    def forward(self, x):
        return self.model(x)


class QNetwork(nn.Module):
    def __init__(self, dim_obs, action_space, tank_encoder, poly_encoder, bullet_encoder, device, embed_dim = 8):
        super().__init__()

        self.dev = device

        self.nvec = action_space["d"].nvec
        self.dim_d = len(self.nvec)
        self.dim_c = int(action_space["c"].shape[0])
        self.dim_act = self.dim_d * embed_dim + self.dim_c

        # embedder
        self.embeds = nn.ModuleList([
            nn.Embedding(n, embed_dim) for n in self.nvec
        ])

        self.layers1 = FCFF(dim_obs + self.dim_act, 1)
        self.layers2 = FCFF(dim_obs + self.dim_act, 1)

        self.te = tank_encoder
        self.pe = poly_encoder
        self.be = bullet_encoder

    def encode_discrete(self, a_d):
        # a_d: (batch_size, n) with integer entries
        embedded = []
        for i in range(self.dim_d):
            col = a_d[:, i]
            embedded.append(self.embeds[i](col)) # (batch_size, embed_dim)
        return torch.cat(embedded, dim = -1) # (batch_size, n * embed_dim)

    def forward(self, obs_batch, act):
        obs_self = torch.tensor(
            np.array([obs["self"] for obs in obs_batch]), device = self.dev
        )
        obs_tank = self.te([obs["tanks"] for obs in obs_batch])
        obs_poly = self.pe([obs["polygons"] for obs in obs_batch])
        obs_bullet = self.be([obs["bullets"] for obs in obs_batch])

        act_d = self.encode_discrete(act["d"])
        act_c = act["c"]
        act_all = torch.cat([act_d, act_c], dim = -1)

        x = torch.cat([obs_self, obs_tank, obs_poly, obs_bullet, act_all], dim = -1)
        return (self.layers1(x), self.layers2(x))


class PNetwork(nn.Module):
    def __init__(self, dim_obs, action_space, tank_encoder, poly_encoder, bullet_encoder, device):
        super().__init__()

        self.dev = device

        space_disc = action_space["d"]
        space_cont = action_space["c"]

        self.n_discrete = space_disc.nvec  # array: [3, 3, 2, 9]
        self.dim_cont = space_cont.shape[0]

        # shared rep
        self.shared = FCFF(dim_obs, HIDDEN_LAYER)

        # discrete
        self.disc_heads = nn.ModuleList([
            nn.Linear(HIDDEN_LAYER, n) for n in self.n_discrete
        ])

        # continuous
        self.cont_heads = FCFF(HIDDEN_LAYER, 2 * self.dim_cont)

        self.te = tank_encoder
        self.pe = poly_encoder
        self.be = bullet_encoder

        self.action_scale = torch.FloatTensor((space_cont.high - space_cont.low) / 2.)
        self.action_bias  = torch.FloatTensor((space_cont.high + space_cont.low) / 2.)


    def forward(self, obs_batch):
        if type(obs_batch) is not list:
            obs_batch = [obs_batch]

        obs_self   = torch.tensor(
            np.array([obs["self"] for obs in obs_batch]), device = self.dev
        )
        obs_tank   = self.te([obs["tanks"] for obs in obs_batch])
        obs_poly   = self.pe([obs["polygons"] for obs in obs_batch])
        obs_bullet = self.be([obs["bullets"] for obs in obs_batch])

        x = torch.cat([obs_self, obs_tank, obs_poly, obs_bullet], dim = -1)
        x = self.shared(x)

        # discrete output
        logits_d   = [head(x) for head in self.disc_heads]

        # continuous output
        output     = self.cont_heads(x)
        mean, lstd = torch.chunk(output, chunks = 2, dim = -1)

        lstd = torch.clamp(lstd, LOG_STD_MIN, LOG_STD_MAX)
        return logits_d, mean, lstd


    def sample(self, obs):
        if type(obs) is not list:
            obs = [obs]

        logits_d, mean, lstd = self.forward(obs)
        std = torch.exp(lstd)

        # discrete
        d_dists = [torch.distributions.Categorical(logits=logit) for logit in logits_d]
        d_actions = [dist.sample() for dist in d_dists]
        d_log_probs = [dist.log_prob(a) for dist, a in zip(d_dists, d_actions)]

        # continuous
        N     = torch.distributions.Normal(mean, std)
        x     = N.rsample()
        tx    = torch.tanh(x)
        act_c = tx * self.action_scale + self.action_bias

        log_prob = N.log_prob(x) - torch.log(self.action_scale * (1 - tx.pow(2)) + 1e-7)
        log_prob = log_prob.sum(1, keepdim = True)

        # Combine log probs
        total_log_prob = torch.stack(d_log_probs, dim = 1).sum(dim = 1, keepdim = True) + log_prob

        best_c = torch.tanh(mean) * self.action_scale + self.action_bias

        return {
            "c": act_c,
            "d": torch.stack(d_actions, dim = 1),
        }, total_log_prob, {
            "c": best_c,
            "d": torch.stack([torch.argmax(l, dim = -1) for l in logits_d], dim=1),
        }