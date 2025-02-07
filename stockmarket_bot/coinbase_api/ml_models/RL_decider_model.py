import logging
from typing import Any, Dict, List, Tuple
import gymnasium as gym
from gymnasium import spaces
from coinbase_api.ml_models.data_handlers.abstract_data_handler import AbstractDataHandler
import numpy as np
import numpy.typing as npt

from coinbase_api.models.models import AbstractOHLCV
from ..constants import crypto_models, crypto_features, crypto_predicted_features, crypto_extra_features

# Configure logging
logging.basicConfig(
    filename='training_log.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

class CustomEnv(gym.Env):
    def __init__(self, data_handler: AbstractDataHandler, transaction_cost_factor=1.0) -> None:
        super(CustomEnv, self).__init__()
        self.crypto_models = crypto_models
        self.data_handler = data_handler
        self.transaction_cost_factor = transaction_cost_factor
        N = 1 #! Currently set to handle one crypto currency at a time
        self.action_space = spaces.Box(low=-1, high=1, shape=(N,), dtype=np.float32)  # where N is the number of cryptocurrencies
        # M = len(self.get_crypto_features()) + len(self.get_crypto_predicted_features()) + len(self.get_extra_features())
        M = len(self.get_crypto_features()) + len(self.get_crypto_predicted_features())
        shape_value = M * N + 3  # +1 each because of total volume held and USDC value held and volume of the currency held
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape_value,), dtype=np.float64)

    def set_currency(self, new_currency: AbstractOHLCV):
        self.data_handler.set_currency(new_currency)

    def step(self, action) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        next_state, cost_for_action, terminated, info = self.data_handler.update_state(action)
        self.state = next_state
        self.cost_for_action = cost_for_action
        total_volume = next_state[0]
        reward_q = self.data_handler.get_reward(action)
        self.reward = reward_q
        truncated = False

        if total_volume < self.data_handler.initial_volume / 10:
            terminated = True
            logging.debug(f'Terminated: {terminated}')
            logging.debug(f'Total time steps: {self.data_handler.total_steps}')
            logging.debug(f'Initial timestamp: {self.data_handler.initial_timestamp}')
            logging.debug(f'Current timestamp: {self.data_handler.timestamp}')
            logging.debug(f'Step number: {self.data_handler.step_count}')
            logging.debug(f'Crypto values: {self.data_handler.account_holdings}')
            logging.debug(f'USDC value: {self.data_handler.usdc_held}')
            logging.debug(f'Action: {action}')
            logging.debug(f'Next state: {next_state}')
            logging.debug(f'Reward: {reward_q}')
            logging.debug(f'Total volume: {total_volume}')
            logging.debug(f'Cost for action: {cost_for_action}')
        
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
