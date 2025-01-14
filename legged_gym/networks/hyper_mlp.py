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

    def forward(self, x, hyper_x=None):
        if isinstance(x, dict):
            x = torch.cat([v.flatten(1) for k, v in x.items() if "ids" not in k], dim=-1)
        return self.mlp(x)


class HyperMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hyper_input_dim, hidden_dims=[256, 256, 256], hyper_hidden_dims=[256], hyper_input='id', activation='elu', **kwargs):
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

        if hyper_input == "id":
            self.embed = nn.Embedding(hyper_input_dim, hyper_hidden_dims[0])
        elif hyper_input == "offset":
            self.embed = nn.Sequential(
                nn.Linear(hyper_input_dim, hyper_hidden_dims[0]), activation
            )
        layers = []
        for l in range(len(hyper_hidden_dims) - 1):
            layers.append(nn.Linear(hyper_hidden_dims[l], hyper_hidden_dims[l + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hyper_hidden_dims[-1], output_dim * self.hidden_dim + output_dim))
        self.params = nn.Sequential(*layers)
        self.apply(self.reset_parameter)

    @staticmethod
    def reset_parameter(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)

    def forward(self, x, hyper_x):
        if isinstance(x, dict):
            x = torch.cat([v.flatten(1) for k, v in x.items()], dim=-1)
        feature = self.feature(x)
        params = self.params(self.embed(hyper_x))
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
                        num_robots,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        hyper_hidden_dims=[256],
                        hyper_actor_mean=True,
                        hyper_actor_std=False,
                        hyper_critic=False,
                        hyper_input="id",
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("MLPAC.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(HyperMLPAC, self).__init__()

        hyper_input_dim = num_robots
        actor_obs_shape = {k: np.prod(v) for k, v in actor_obs_shape.items()}
        actor_obs_shape.pop('ids')
        if hyper_input == "offset" and hyper_actor_mean:
            offset_dim = actor_obs_shape.pop('offset')
            hyper_input_dim = offset_dim
        mlp_input_dim_a = np.sum(list(actor_obs_shape.values()))

        critic_obs_shape = {k: np.prod(v) for k, v in critic_obs_shape.items()}
        critic_obs_shape.pop('ids')
        if hyper_input == "offset" and hyper_critic:
            critic_obs_shape.pop('offset')
        mlp_input_dim_c = np.sum(list(critic_obs_shape.values()))


        # Policy
        if hyper_actor_mean:
            self.actor = HyperMLP(mlp_input_dim_a, num_actions, hyper_input_dim, actor_hidden_dims, hyper_hidden_dims, hyper_input, activation)
        else:
            self.actor = MLP(mlp_input_dim_a, num_actions, actor_hidden_dims, activation)

        # Value function
        if hyper_critic:
            self.critic = HyperMLP(mlp_input_dim_c, 1, hyper_input_dim, critic_hidden_dims, hyper_hidden_dims, hyper_input, activation)
        else:
            self.critic = MLP(mlp_input_dim_c, 1, critic_hidden_dims, activation)

        # print(f"Actor MLP: {self.actor}")
        # print(f"Critic MLP: {self.critic}")

        # Action noise
        if hyper_actor_std:
            self.std = nn.Parameter(init_noise_std * torch.ones((num_robots, num_actions)))
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.hyper_actor_mean = hyper_actor_mean
        self.hyper_critic = hyper_critic
        self.hyper_input = hyper_input
        
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
        hyper_x = observations["ids"].squeeze(-1).long()
        x = {k: v for k, v in observations.items() if k != "ids"}
        if self.hyper_input == "offset" and self.hyper_actor_mean:
            hyper_x = observations["offset"].flatten(1)
            x = {k: v for k, v in x.items() if k != "offset"}
        mean = self.actor(x, hyper_x)
        if len(self.std.shape) == 1:
            std = self.std
        else:
            ids = observations['ids'].squeeze(-1).long()
            std = self.std[ids]
        self.distribution = Normal(mean, mean*0. + std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        hyper_x = observations["ids"].squeeze(-1).long()
        x = {k: v for k, v in observations.items() if k != "ids"}
        if self.hyper_input == "offset" and self.hyper_actor_mean:
            hyper_x = observations["offset"].flatten(1)
            x = {k: v for k, v in x.items() if k != "offset"}
        actions_mean = self.actor(x, hyper_x)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        hyper_x = critic_observations["ids"].squeeze(-1).long()
        x = {k: v for k, v in critic_observations.items() if k != "ids"}
        if self.hyper_input == "offset" and self.hyper_critic:
            hyper_x = critic_observations["offset"].flatten(1)
            x = {k: v for k, v in x.items() if k != "offset"}
        value = self.critic(x, hyper_x)
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
