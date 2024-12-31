# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256, 256], activation='elu', **kwargs):
        if kwargs:
            print("MLP.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(MLP, self).__init__()

        activation = get_activation(activation)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, dict):
            x = torch.cat([v.flatten(1) for k, v in x.items() if "ids" not in k], dim=-1)
        return self.mlp(x)


class HyperMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_ids, hidden_dims=[256, 256, 256], hyper_hidden_dims=[256], activation='elu', **kwargs):
        if kwargs:
            print("MLP.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(HyperMLP, self).__init__()
        self.hidden_dim = hidden_dims[-1]
        self.output_dim = output_dim

        activation = get_activation(activation)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
            layers.append(activation)
        self.feature = nn.Sequential(*layers)

        self.embed = nn.Embedding(num_ids, self.hidden_dim)
        layers = []
        hyper_hidden_dims = [self.hidden_dim] + hyper_hidden_dims + [self.output_dim * self.hidden_dim + self.output_dim]
        for l in range(len(hyper_hidden_dims) - 1):
            layers.append(nn.Linear(hyper_hidden_dims[l], hyper_hidden_dims[l + 1]))
            if l < len(hyper_hidden_dims) - 2:
                layers.append(activation)
        self.params = nn.Sequential(*layers)

    def forward(self, x):
        ids = x['ids'].long()
        if isinstance(x, dict):
            x = torch.cat([v.flatten(1) for k, v in x.items() if "ids" not in k], dim=-1)
        feature = self.feature(x)
        params = self.params(self.embed(ids))
        weights, bias = torch.split(params, [self.output_dim * self.hidden_dim, self.output_dim], dim=-1)
        weights = weights.view(-1, self.output_dim, self.hidden_dim)
        bias = bias.view(-1, self.output_dim)
        out = torch.bmm(weights, feature.unsqueeze(-1)).squeeze(-1) + bias
        return out


class HyperMLPAC(nn.Module):
    is_recurrent = False
    def __init__(self,  actor_obs_shape,
                        critic_obs_shape,
                        num_actions,
                        num_ids,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        hyper_hidden_dims=[256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("MLPAC.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(HyperMLPAC, self).__init__()

        mlp_input_dim_a = np.sum([np.prod(val) for key, val in actor_obs_shape.items() if "ids" not in key])
        mlp_input_dim_c = np.sum([np.prod(val) for key, val in critic_obs_shape.items() if "ids" not in key])

        # Policy
        self.actor = HyperMLP(mlp_input_dim_a, num_actions, num_ids, actor_hidden_dims, hyper_hidden_dims, activation)

        # Value function
        self.critic = MLP(mlp_input_dim_c, 1, critic_hidden_dims, activation)

        # print(f"Actor MLP: {self.actor}")
        # print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
