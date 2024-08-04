from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from torch import nn

import torch as th

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO


class CustomNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.Snake_lstm = nn.LSTM(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
            nn.Linear(128,10)
        )

        feature_dim = feature_dim - 800 + 10

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
            nn.Linear(128,last_layer_dim_pi)
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        last_30 = features[:,:800].reshape(-1, 400,2)
        x, _ = self.Snake_lstm(last_30)
        x1 = self.lstm_dnn(x)
        model_input =  th.hstack([x1, features[:,800:]])
        return self.policy_net(model_input)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        last_30 = features[:,:800].reshape(-1, 400,2)
        x, _ = self.Snake_lstm(last_30)
        x1 = self.lstm_dnn(x)
        model_input =  th.hstack([x1, features[:,800:]])
        return self.value_net(model_input)

class model1(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


