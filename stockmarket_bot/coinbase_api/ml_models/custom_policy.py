
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch.optim import Adam

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU, *args, **kwargs):
        if net_arch is None:
            net_arch = [64, 64]  # Default architecture with two hidden layers of 64 units each

        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs)
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch, activation_fn)
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_space.shape[0])
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = Adam(self.parameters(), lr=lr_schedule(1.0), weight_decay=1e-4)  # Adding L2 regularization here

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        value = self.value_net(latent_vf)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, value, log_prob

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
