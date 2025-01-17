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


class CustomNetwork2(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,10)
        )
        feature_dim = feature_dim - 800 + 10
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_2(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork2(self.features_dim)



class CustomNetwork3(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,10)
        )
        feature_dim = feature_dim - 800 + 10
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
            nn.Linear(128,last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_3(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork3(self.features_dim)


class CustomNetwork4(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,10)
        )
        feature_dim = feature_dim - 800 + 10
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_4(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork4(self.features_dim)



class CustomNetwork5(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,10)
        )
        feature_dim = feature_dim - 800 + 10
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_5(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork5(self.features_dim)


class CustomNetwork6(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,10)
        )
        feature_dim = feature_dim - 800 + 10
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256,256), 
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256,256), 
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_6(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork5(self.features_dim)


class CustomNetwork7(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,10)
        )
        feature_dim = feature_dim - 800 + 10
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Mish(),
            nn.Linear(256,128), nn.Mish(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_3_mish(ActorCriticPolicy):
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
            *args,
            **kwargs,
        )
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork7(self.features_dim)



class CustomNetwork8(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,10)
        )
        feature_dim = feature_dim - 800 + 10
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_3_lkrelu(ActorCriticPolicy):
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
            *args,
            **kwargs,
        )
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork8(self.features_dim)




class CustomNetwork5_relu(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,10)
        )
        feature_dim = feature_dim - 800 + 10
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,128), nn.LeakyReLU(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_5_relu(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork5_relu(self.features_dim)

class CustomNetwork5_tanh(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
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
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_5_tanh(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork5_tanh(self.features_dim)




class CustomNetwork5_softsign(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.Softsign(),
            nn.Linear(256,256), nn.Softsign(),
            nn.Linear(256,256), nn.Softsign(),
            nn.Linear(256,256), nn.Softsign(),
            nn.Linear(256,128), nn.Softsign(),
            nn.Linear(128,10)
        )
        feature_dim = feature_dim - 800 + 10
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Softsign(),
            nn.Linear(256,256), nn.Softsign(),
            nn.Linear(256,256), nn.Softsign(),
            nn.Linear(256,256), nn.Softsign(),
            nn.Linear(256,128), nn.Softsign(),
            nn.Linear(128,last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Softsign(),
            nn.Linear(256,256), nn.Softsign(),
            nn.Linear(256,256), nn.Softsign(),
            nn.Linear(256,256), nn.Softsign(),
            nn.Linear(256,128), nn.Softsign(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_5_softsign(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork5_softsign(self.features_dim)



class CustomNetwork3_Softsign(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.Snake_lstm = nn.GRU(2,2,4)
        self.lstm_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800,256), nn.Softsign(),
            nn.Linear(256,128), nn.Softsign(),
            nn.Linear(128,10)
        )
        feature_dim = feature_dim - 800 + 10
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Softsign(),
            nn.Linear(256,128), nn.Softsign(),
            nn.Linear(128,last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Softsign(),
            nn.Linear(256,128), nn.Softsign(),
            nn.Linear(128,last_layer_dim_pi)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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

class model1_3_Softsign(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork3_Softsign(self.features_dim)

