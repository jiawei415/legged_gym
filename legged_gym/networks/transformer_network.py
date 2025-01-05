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
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.distributions import Normal


class ResidualBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1, dim2)
        self.activation = nn.GELU()
        self.fc2 = torch.nn.Linear(dim2, dim2)
    def forward(self, x):
        hidden = self.fc1(x)
        residual = hidden
        hidden = self.activation(hidden)
        out = self.fc2(hidden)
        out += residual
        return out


class MLPBlock(nn.Module):
    def __init__(self, dim1, dim2, hidden, activation='elu'):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1, hidden)
        self.activation = get_activation(activation)
        self.fc2 = torch.nn.Linear(hidden, dim2)
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        state_shape: Dict[str, tuple],
        num_layers: int = 2,
        num_heads: int = 1,
        embedding_dim: int = 32,
        mlp_embedding: bool = False,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        enable_pos_embed: bool = True,
    ):
        super().__init__()
        seq_len = state_shape['link_obs'][0] + 2
        root_dim = state_shape['root_obs'][-1]
        link_dim = state_shape['link_obs'][-1]
        cmd_dim = state_shape['cmd_obs'][-1]
        map_dim = state_shape['map_obs'][-1] if 'map_obs' in state_shape else 0
        offset_dim = state_shape['offset'][-1] if 'offset' in state_shape else 0


        # additional seq_len embeddings for padding timesteps
        if mlp_embedding:
            self.link_emb = ResidualBlock(link_dim + offset_dim, embedding_dim)
            self.root_emb = ResidualBlock(root_dim, embedding_dim)
            self.cmd_emb = ResidualBlock(cmd_dim + map_dim, embedding_dim)
        else:
            self.link_emb = nn.Linear(link_dim + offset_dim, embedding_dim)
            self.root_emb = nn.Linear(root_dim, embedding_dim)
            self.cmd_emb = nn.Linear(cmd_dim + map_dim, embedding_dim)

        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.emb_drop = nn.Dropout(embedding_dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(embedding_dim)
        # self.action_head = nn.Linear(embedding_dim * 3, 1)

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.enable_pos_embed = enable_pos_embed

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def positional_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        """Create absolute positional embedding

        inputs : shape (batch_size, seq_maxlen, dims)

        Returns:
            position_embedding : shape (batch_size, seq_maxlen, pos_dim)
        """

        device = inputs.device
        seq_len, embedding_dim = inputs.size(1), inputs.size(2)

        # Compute the position frequencies
        position_j = 1. / torch.pow(10000., 2 * torch.arange(embedding_dim // 2, dtype=torch.float32, device=device) / embedding_dim)
        position_j = position_j.unsqueeze(0)  # shape (1, embedding_dim//2)

        # Compute the sequence of indices
        position_i = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # shape (seq_len, 1)

        # Compute the product of indices and frequencies
        position_ij = torch.matmul(position_i, position_j)  # shape (seq_len, embedding_dim//2)

        # Concatenate sine and cosine functions of positional encodings
        position_ij = torch.cat([torch.cos(position_ij), torch.sin(position_ij)], dim=1)  # shape (seq_len, embedding_dim)

        # Expand the positional encodings to match the input batch size and sequence length
        position_embedding = position_ij.unsqueeze(0).repeat(inputs.size(0), 1, 1)  # shape (batch_size, seq_len, embedding_dim)

        return position_embedding

    def forward(
        self,
        states: Dict[str, torch.Tensor],  # [batch_size, seq_len, state_dim]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        link_obs = states['link_obs']
        if 'offset' in states:
            offset = states['offset']
            link_obs = torch.cat([link_obs, offset], dim=-1)
        root_obs = states['root_obs']
        cmd_obs = states['cmd_obs']
        if 'map_obs' in states:
            map_obs = states['map_obs']
            cmd_obs = torch.cat([cmd_obs, map_obs], dim=-1)

        # [batch_size, seq_len, emb_dim]
        link_emb = self.link_emb(link_obs)
        root_emb = self.root_emb(root_obs).unsqueeze(1)
        cmd_emb = self.cmd_emb(cmd_obs).unsqueeze(1)
        state_emb = torch.cat([cmd_emb, root_emb, link_emb], dim=1)
        if self.enable_pos_embed:
            state_pos = self.positional_embedding(state_emb)
            state_emb = state_emb + state_pos

        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        state_emb = self.emb_norm(state_emb)
        state_emb = self.emb_drop(state_emb)

        for block in self.blocks:
            state_emb = block(state_emb, padding_mask=padding_mask)

        state_emb = self.out_norm(state_emb)
        return state_emb[:, 2:, :]


class Actor(nn.Module):
    def __init__(
        self,
        process_net: nn.Module,
        action_dim: int,
        mlp_prediction: bool = False,
        activation: str = 'elu',
        token_type: str = 'last',
    ):
        super().__init__()
        self.process_net = process_net
        embedding_dim = process_net.embedding_dim
        output_dim = action_dim if token_type != 'all' else 1
        if mlp_prediction:
            self.action_head = MLPBlock(embedding_dim, output_dim, embedding_dim, activation)
        else:
            self.action_head = nn.Linear(embedding_dim, output_dim)
        self.token_type = token_type

    def forward(self, states: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        feature = self.process_net(states)
        if self.token_type == 'last':
            feature = feature[:, -1]
        elif self.token_type == 'mean':
            feature = torch.mean(feature, dim=1)
        elif self.token_type == 'sum':
            feature = torch.sum(feature, dim=1)
        elif self.token_type == 'all':
            feature = feature
        else:
            raise ValueError(f"Unknown token_type: {self.token_type}")
        out = self.action_head(feature)
        if self.token_type == 'all':
            out = out.squeeze(-1)
        return out


class Critic(nn.Module):
    def __init__(
        self,
        process_net: nn.Module,
        mlp_prediction: bool = False,
        activation: str = 'elu',
        token_type: str = 'last',
    ):
        super().__init__()
        self.process_net = process_net
        embedding_dim = process_net.embedding_dim
        if mlp_prediction:
            self.value_head = MLPBlock(embedding_dim, 1, embedding_dim, activation)
        else:
            self.value_head = nn.Linear(embedding_dim, 1)
        self.token_type = token_type

    def forward(self, states: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        feature = self.process_net(states)
        if self.token_type == 'last':
            feature = feature[:, -1]
        elif self.token_type == 'mean':
            feature = torch.mean(feature, dim=1)
        elif self.token_type == 'sum':
            feature = torch.sum(feature, dim=1)
        elif self.token_type == 'all':
            feature = feature
        else:
            raise ValueError(f"Unknown token_type: {self.token_type}")
        out = self.value_head(feature)
        if self.token_type == 'all':
            out = torch.mean(out, dim=1)
        return out


class TransformerAC(nn.Module):
    is_recurrent = False
    def __init__(self,  actor_obs_shape,
                        critic_obs_shape,
                        num_actions,
                        num_layers: int = 2,
                        num_heads: int = 1,
                        embedding_dim=32,
                        shared_backbone=True,
                        mlp_embedding=False,
                        mlp_prediction=False,
                        enable_pos_embed=True,
                        actor_token_type='last',
                        critic_token_type='last',
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("TransformerAC.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(TransformerAC, self).__init__()

        # activation = get_activation(activation)
        process_net = Transformer(
            actor_obs_shape, num_layers, num_heads, embedding_dim, mlp_embedding, enable_pos_embed=enable_pos_embed)

        # Policy
        self.actor = Actor(process_net, num_actions, mlp_prediction, activation, actor_token_type)
        # self.actor = Transformer(actor_obs_shape)

        # Value function
        if not shared_backbone:
            process_net = Transformer(critic_obs_shape, num_layers, num_heads, embedding_dim, mlp_embedding)
        self.critic = Critic(process_net, mlp_prediction, activation, critic_token_type)
        # self.critic = Transformer(critic_obs_shape, output_value=True)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
        self.actor.action_head.apply(self._init_weights)
        self.critic.value_head.apply(self._init_weights)

    # @staticmethod
    # not used at the moment
    # def init_weights(sequential, scales):
    #     [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
    #      enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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
