import torch
import torch.nn as nn
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from torch.optim import Adam
from gymnasium import spaces
from typing import Callable, Optional, Tuple
from sb3_contrib.common.recurrent.type_aliases import RNNStates

class CustomRecurrentPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Callable[[float], float],
                 net_arch=None,
                 activation_fn=nn.ReLU,
                 lstm_hidden_size=128,
                 lstm_num_layers=1,
                 n_envs=1,
                 *args, **kwargs):
        # Use a default network structure if none is provided:
        if net_arch is None:
            net_arch = {
                "shared": [256, 256, 128],
                "pi": [128, 64, 32],
                "vf": [128, 64, 32]
            }
        self.n_envs = n_envs
        self.net_arch = net_arch
        super(CustomRecurrentPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, lstm_hidden_size=lstm_hidden_size, n_lstm_layers=lstm_num_layers, *args, **kwargs)
        
        # Build a custom shared network using net_arch["shared"]
        shared_arch = net_arch["shared"]
        layers = []
        input_dim = self.features_dim  # e.g. 14 from your observation space
        for hidden_dim in shared_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            input_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)  # Output dimension = shared_arch[-1] (e.g. 128)
        
        # LSTM layer: its input size must match the shared net's output dimension.
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=net_arch["shared"][-1],  # expected to be 128
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        # Expose the LSTM as "lstm_actor" so RecurrentPPO can access it.
        self.lstm_actor = self.lstm
        
        # Build separate MLP heads for policy and value.
        self.pi_mlp = self._build_mlp(self.lstm_hidden_size, net_arch["pi"], activation_fn)
        self.vf_mlp = self._build_mlp(self.lstm_hidden_size, net_arch["vf"], activation_fn)
        
        # Final output layers.
        final_pi_dim = net_arch["pi"][-1]
        final_vf_dim = net_arch["vf"][-1]
        self.action_net = nn.Linear(final_pi_dim, action_space.shape[0])
        self.value_net = nn.Linear(final_vf_dim, 1)
        
        # Set up optimizer.
        self.optimizer = Adam(self.parameters(), lr=lr_schedule(1.0), weight_decay=1e-4)
        
    def _build_mlp(self, input_dim, layer_sizes, activation_fn):
        layers = []
        for size in layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(activation_fn())
            input_dim = size
        return nn.Sequential(*layers)
    
    def forward(self, obs, lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                episode_starts: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Expects obs of shape (batch, sequence, features). 
        lstm_states is a tuple (h, c) with shape (num_layers, batch, lstm_hidden_size).
        episode_starts is a tensor of shape (batch,) of floats indicating whether an episode has restarted.
        Returns: actions, value estimates, log probabilities, and updated lstm_states.
        """
        obs = obs.to(torch.float32)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, features)
        batch_size, seq_length, _ = obs.shape
        
        # Flatten obs to (batch*sequence, features)
        flat_obs = obs.view(-1, obs.shape[-1])
        shared_out = self.shared_net(flat_obs)  # shape: (batch*sequence, shared_dim)
        shared_out = shared_out.view(batch_size, seq_length, -1)  # (batch, sequence, shared_dim)
        
        # Initialize LSTM states if not provided.
        if lstm_states is not None and hasattr(lstm_states, "pi"):
            lstm_states = lstm_states.pi  # now lstm_states is a tuple (h, c)
        if lstm_states is None:
            h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm_hidden_size, device=obs.device)
            c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm_hidden_size, device=obs.device)
            lstm_states = (h0, c0)
        else:
            h0, c0 = lstm_states
            if episode_starts is not None:
                mask = episode_starts.to(torch.bool)
                if mask.any():
                    h0 = h0.clone()
                    c0 = c0.clone()
                    h0[:, mask, :] = 0.0
                    c0[:, mask, :] = 0.0
                lstm_states = (h0, c0)
        # If episode_starts is provided, reset LSTM states for episodes that have just started.
        # if episode_starts is not None:
        #     # episode_starts is a tensor of shape (batch,)
        #     mask = episode_starts.to(torch.bool)
        #     if mask.any():
        #         h, c = lstm_states
        #         h = h.clone()  # clone to get a mutable copy
        #         c = c.clone()
        #         h[:, mask, :] = 0.0
        #         c[:, mask, :] = 0.0
        #         lstm_states = (h, c)

        
        lstm_output, new_lstm_states  = self.lstm(shared_out, lstm_states)
        # Use the output from the last timestep.
        last_output = lstm_output[:, -1, :]
        
        # Policy branch.
        pi_latent = self.pi_mlp(last_output)
        distribution = self._get_action_dist_from_latent(pi_latent)
        # Value branch.
        vf_latent = self.vf_mlp(last_output)
        value = self.value_net(vf_latent)
        
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        new_state = RNNStates(pi=new_lstm_states, vf=new_lstm_states)
        return actions, value, log_prob, new_state

    def _get_action_dist_from_latent(self, latent: torch.Tensor):
        mean_actions = self.action_net(latent)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
