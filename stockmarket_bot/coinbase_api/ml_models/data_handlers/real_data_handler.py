from typing import Any, Dict, Tuple
import numpy as np
import numpy.typing as npt
from coinbase_api.enums import Actions
from coinbase_api.ml_models.data_handlers.abstract_data_handler import AbstractDataHandler
from coinbase_api.models.models import AbstractOHLCV
from datetime import datetime
from coinbase_api.utilities.cb_provider import CbProvider

cb_provider = CbProvider()

class RealDataHandler(AbstractDataHandler):
    def __init__(self, crypto: AbstractOHLCV, initial_volume = 0) -> None:
        self.crypto = crypto
        self.wallets_dict = cb_provider.get_wallets()
        self.initial_volume = cb_provider.get_total_volume()
        self.total_volume = self.initial_volume
        
        self.action_factor = 0.5 #TODO action factor mapping?
        self.timestamp = datetime.now()
        self.account_holding = cb_provider.get_account_holding(self.crypto)
        
    def update_state(self, action) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        self.costs_for_action = self.cost_for_action(action)
        done = False
        info = {}
        self.next_state = self.get_current_state()
        return self.next_state, self.costs_for_action, done, info
    
    def reset_state(self) -> npt.NDArray[np.float16]:
        self.wallets_dict = cb_provider.get_wallets()
        return self.get_current_state()
    
    def get_current_state(self):
        self.total_volume = cb_provider.get_total_volume()
        self.account_holding = cb_provider.get_account_holding(self.crypto)
        self.new_crypto_data = cb_provider.get_crypto_data(self.crypto)
        self.usdc_held = cb_provider.get_liquidity()
        return np.array([self.total_volume, self.usdc_held, self.account_holding] + self.new_crypto_data)
    
    # def set_currency(self, new_currency: AbstractOHLCV, verbose=True):
    #     self.crypto = new_currency
    #     self.account_holdings = self.get_account_holdings()
    #     self.state = self.get_current_state()
        
    # def get_product_types(self):
    #     return [f"{crypto.symbol}-USDC" for crypto in crypto_models]
    
    # def calculate_total_volume(self):
    #     request_body = {
    #         "get_all_products": "true",
    #     }
    #     all_products = api_request_with_auth(ApiPath.PRODUCTS.value, request_method=Method.GET, request_body=request_body)
    #     products = all_products["products"]
    #     usable_products = [item for item in products if item["product_id"] in self.get_product_types()]
    #     products_reduced = {item["product_id"]: item["price"] for item in usable_products}
    #     total = sum([float(val) * float(products_reduced[f"{key}-USDC"]) for key, val in self.wallets_dict.items() if f"{key}-USDC" in products_reduced.keys()])
    #     total_with_usdc = total + float(self.wallets_dict["USDC"])
    #     return total_with_usdc
    
    # def get_account_holdings(self):
    #     return [float(self.wallets_dict[crypto_model.symbol]) if crypto_model.symbol in self.wallets_dict.keys() else 0.0 for crypto_model in crypto_models]
    
    # def get_liquidity(self) -> float:
    #     return float(self.wallets_dict['USDC'])
            
    # def get_new_crypto_data(self):
    #     all_entries = []
    #     start = datetime.now() - timedelta(minutes=5*50)
    #     end = datetime.now() + timedelta(minutes=1)
    #     raw_data = self.get_crypto_raw_candles(start, end, self.crypto.symbol)
    #     crypto_instances = self.parse_raw_data(self.crypto, raw_data["candles"])
    #     crypto_instance_now: AbstractOHLCV = self.add_calculated_parameters(crypto_instances)
    #     all_entries = all_entries + self.crypto_to_list(crypto_instance_now)
    #     return all_entries

    # def get_crypto_raw_candles(self, start, end, crypto_symbol):
    #     raw_data = api_request_with_auth(
    #         request_path=f"{ApiPath.PRODUCTS.value}/{crypto_symbol}-USDC/candles",
    #         request_method=Method.GET,
    #         request_body = {
    #             "start" : str(int(datetime.timestamp(start))),
    #             "end": str(int(datetime.timestamp(end))),
    #             "granularity": Granularities.FIVE_MINUTE.value
    #         })
            
    #     return raw_data
    
    # def parse_raw_data(self, crypto_model: AbstractOHLCV, data) -> List[AbstractOHLCV]:
    #     try:
    #         entries = []
    #         for item in data:
    #             try:
    #                 timestamp = float(item["start"])
    #                 low = float(item["low"])
    #                 high = float(item["high"])
    #                 opening = float(item["open"])
    #                 close_base = float(item["close"])
    #                 volume = float(item["volume"])
                    
    #                 new_entry = crypto_model(
    #                     timestamp=datetime.fromtimestamp(int(timestamp), tz=timezone.utc),
    #                     open=opening,
    #                     high=high,
    #                     low=low,
    #                     close=close_base,
    #                     volume=volume,
    #                 )
    #                 entries.append(new_entry)
    #             except Exception as e:
    #                 print(f'Encountered an error with item {item}: {e}')
    #         return entries
    #     except Exception as e:
    #         print(f'Encountered the following error: {e}')
    
    def cost_for_action(self, action):
        print('TODO: Implement cost for action method')
        return 0
    
    # def get_crypto_features(self) -> List[str]:
    #     return crypto_features

    # def get_crypto_predicted_features(self) -> List[str]:
    #     return crypto_predicted_features
    
    # def crypto_to_list(self, crypto: AbstractOHLCV) -> List[float]:
    #     attributes = [getattr(crypto, fieldname)for fieldname in self.get_crypto_features()]
    #     return [attr if attr != None else 0.0 for attr in attributes]
    
    def get_step_count(self) -> int:
        #! needs implementation
        return 0
    
    def get_current_state_output(self, action) -> str:
        return "No Output for current state as of yet in real data output"
    
    def get_total_steps(self) -> int:
        #! needs implementation
        return 0
    
    def get_reward(self, action: Actions)-> float:
        reward = ((self.total_volume / self.initial_volume) - 1)*3
        cost_for_action = self.cost_for_action(action)
        return reward - cost_for_action*0.25 #! not full scope since this would scale awkwardly. Cost = 4 usdc = -4, which is not reachable the other way around