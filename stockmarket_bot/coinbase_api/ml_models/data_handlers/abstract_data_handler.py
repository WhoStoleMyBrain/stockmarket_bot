
from typing import Any, Dict, List, Tuple
import numpy as np
import numpy.typing as npt
from coinbase_api.models.models import AbstractOHLCV, Account
from datetime import datetime
class AbstractDataHandler:
    def __init__(self, initial_volume) -> None:
        raise NotImplementedError
    
    def get_initial_crypto_prices(self) -> Dict[str, float]:
        raise NotImplementedError
    
    def get_reward_ratios_for_current_timestep(self) -> Dict[str, float]:
        raise NotImplementedError
    
    def get_earliest_timestamp(self) -> datetime:
        raise NotImplementedError
    
    def get_maximum_timestamp(self) -> datetime:
        raise NotImplementedError
    
    def get_starting_timestamp(self) -> datetime:
        raise NotImplementedError
    
    def map_buy_action(self, buy_action: float, action_factor: float)-> float:
        raise NotImplementedError
    
    def map_sell_action(self, sell_action: float, action_factor: float)-> float:
        raise NotImplementedError
        
    def get_current_state(self) -> npt.NDArray[np.float16]:
        raise NotImplementedError

    def update_state(self, action) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        raise NotImplementedError
    
    def get_liquidity_string(self) -> str:
        raise NotImplementedError
    
    def volume_decay(self, factor:float=0.002) -> None:
        raise NotImplementedError

    def reset_state(self) -> npt.NDArray[np.float16]:
        raise NotImplementedError

    def get_reward_ratios_for_current_timestep(self) -> Dict[str, float]:
        raise NotImplementedError
    
    def get_crypto_account(self, symbol: str) -> Account:
        raise NotImplementedError
    
    def cost_for_action(self, action: List[float]) -> float:
        raise NotImplementedError
    
    def calculate_transaction_volume(self, crypto: AbstractOHLCV, is_buy: bool, factor: float, total_buy_action: float) -> float:
        raise NotImplementedError
    
    def get_crypto_features(self) -> List[str]:
        raise NotImplementedError
    
    def get_crypto_predicted_features(self) -> List[str]:
        raise NotImplementedError
    
    def crypto_to_list(self, crypto: AbstractOHLCV) -> List[float]:
        raise NotImplementedError
    
    def get_new_prediction_data(self, crypto_model:AbstractOHLCV, timestamp:datetime) -> List[float]:
        raise NotImplementedError
    
    def prepare_simulation_database(self) -> None:
        raise NotImplementedError
    
    def reset_account_data(self) -> None:
        raise NotImplementedError
    
    def get_new_instance(self, crypto_model:AbstractOHLCV, instance:AbstractOHLCV) -> AbstractOHLCV:
        raise NotImplementedError
    
    def get_liquidity(self) -> float:
        raise NotImplementedError
    
    def get_account_holdings(self) -> List[float]:
        raise NotImplementedError
    
    def get_new_crypto_data(self) -> List[float]:
        raise NotImplementedError
