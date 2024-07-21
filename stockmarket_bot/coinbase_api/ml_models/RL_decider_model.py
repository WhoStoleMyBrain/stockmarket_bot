from typing import Any, Dict, List, Tuple
import gymnasium as gym
from gymnasium import spaces
from coinbase_api.ml_models.data_handlers.abstract_data_handler import AbstractDataHandler
from coinbase_api.utilities.utils import calculate_total_volume, initialize_default_cryptos
import numpy as np
import numpy.typing as npt
from ..constants import crypto_models, crypto_features, crypto_predicted_features, crypto_extra_features
from ..enums import Actions


class CustomEnv(gym.Env):
    def __init__(self, data_handler:AbstractDataHandler) -> None:
        super(CustomEnv, self).__init__()
        self.crypto_models = crypto_models
        self.data_handler = data_handler
        N = len(self.crypto_models)
        self.action_space = spaces.Box(low=-1, high=1, shape=(N,), dtype=np.float32)  # where N is the number of cryptocurrencies
        M = len(self.get_crypto_features()) + len(self.get_crypto_predicted_features()) + len(self.get_extra_features())
        shape_value = M*N + 2 #! +1 because of total volume held and USDC value held
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape_value,), dtype=np.float64)

    def step(self, action) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        next_state, cost_for_action, terminated, info = self.data_handler.update_state(action)
        # self.next_state = next_state
        self.state = next_state
        total_volume = next_state[0]
        usdc = next_state[1]
        reward_q = self.data_handler.get_reward(action)
        self.reward = reward_q
        truncated = False
        print(self.data_handler.get_current_state_output(action))
        if (total_volume < self.data_handler.initial_volume / 10):
            terminated = True
        return next_state, reward_q, terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[npt.NDArray[np.float16], Dict[Any, Any]]:
        initial_state = self.data_handler.reset_state()
        info = {}
        return initial_state, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_crypto_features(self) -> List[str]:
        return crypto_features

    def get_crypto_predicted_features(self) -> List[str]:
        return crypto_predicted_features
    
    def get_extra_features(self) -> List[str]:
        return crypto_extra_features



    