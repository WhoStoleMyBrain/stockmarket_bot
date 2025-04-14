import torch
import torch.nn as nn
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from torch.optim import Adam
from gymnasium import spaces
from typing import Callable, Optional, Tuple
from sb3_contrib.common.recurrent.type_aliases import RNNStates

class CustomRecurrentPolicyOptimizer(RecurrentActorCriticPolicy):
    def __init__(self, lr_schedule: Callable[[float], float], *args, **kwargs):
        super(CustomRecurrentPolicyOptimizer, self).__init__(*args, **kwargs)
        # Override the default optimizer with a custom Adam optimizer,
        # Set up optimizer.
        self.optimizer = Adam(self.parameters(), lr=lr_schedule(1.0), weight_decay=1e-4)
        