import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch.optim import Adam

class CustomRecurrentPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU, lstm_hidden_size=128, lstm_num_layers = 1, *args, **kwargs):
        # Use a default network structure if none provided
        if net_arch is None:
            # A moderately sized network
            net_arch = {
                "shared": [256, 256, 128],
                "pi": [128, 64, 32],
                "vf": [128, 64, 32]
            }
        self.net_arch = net_arch
        super(CustomRecurrentPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs)
        
        # Build a shared MLP extractor
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch["shared"], activation_fn)
        # self.extract_features = lambda obs: self.mlp_extractor(obs)[0]
        # Add an LSTM layer to capture temporal dependencies
        # Input dimension for LSTM is the output dimension of the shared MLP (for policy branch)
        # lstm_hidden_size = kwargs.get("lstm_hidden_size", 128)
        self.lstm_hidden_size = lstm_hidden_size
        # lstm_num_layers = kwargs.get("lstm_num_layers", 128)
        self.lstm = nn.LSTM(input_size=self.mlp_extractor.latent_dim_pi, hidden_size=self.lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        
        # Policy and Value heads
        self.action_net = nn.Linear(self.lstm_hidden_size, action_space.shape[0])
        self.value_net = nn.Linear(self.lstm_hidden_size, 1)
        
        # Optimizer with Adam and L2 regularization
        self.optimizer = Adam(self.parameters(), lr=lr_schedule(1.0), weight_decay=1e-4)

    def forward(self, obs, deterministic=False, lstm_states=None):
        """
        obs should be shaped as (batch, sequence, features).
        If lstm_states is None, they will be initialized to zeros.
        Returns actions, value estimates, log probabilities, and updated lstm_states.
        """
        obs = obs.to(torch.float32)
        batch_size, seq_length = obs.shape
        # Flatten the observation for feature extraction
        flat_obs = obs.view(-1, obs.shape[-1])
        print("Input obs shape:", obs.shape)
        print("Flat obs shape:", flat_obs.shape)
        features = self.extract_features(flat_obs)
        print("Features shape before reshape:", features.shape)
        features = features.view(batch_size, seq_length, -1)
        print("Features shape after reshape:", features.shape)
        
        # Initialize LSTM states if needed
        if lstm_states is None:
            h0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=features.device)
            c0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=features.device)
            lstm_states = (h0, c0)
            
        lstm_output, lstm_states = self.lstm(features, lstm_states)
        # Use the last output from the LSTM for action and value computation
        last_output = lstm_output[:, -1, :]
        distribution = self._get_action_dist_from_latent(last_output)
        value = self.value_net(last_output)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, value, log_prob, lstm_states

    def _get_action_dist_from_latent(self, latent):
        mean_actions = self.action_net(latent)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
