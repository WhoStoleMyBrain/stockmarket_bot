from typing import Any, Dict, List, Tuple
import numpy as np
import numpy.typing as npt
from coinbase_api.enums import Actions, Database, Granularities
from coinbase_api.ml_models.data_handlers.abstract_data_handler import AbstractDataHandler
from coinbase_api.models.generated_models import ETH
from coinbase_api.models.models import AbstractOHLCV, Account, Prediction
from datetime import datetime, timedelta
from coinbase_api.utilities.prediction_handler import PredictionHandler
from coinbase_api.views.views import cb_auth
from coinbase_api.constants import crypto_models, crypto_features, crypto_predicted_features, crypto_extra_features

class RealDataHandler(AbstractDataHandler):
    def __init__(self, initial_volume = 0) -> None:
        crypto_wallets = cb_auth.restClientInstance.get_accounts()
        #! TODO: Implement more fetching if has_next is true
        self.wallets_dict = {wallet["available_balance"]["currency"]: wallet["available_balance"]["value"] for wallet in crypto_wallets["accounts"]}
        self.initial_volume = self.calculate_total_volume()
        self.total_volume = self.initial_volume
        print(self.wallets_dict)
        self.action_factor = 0.2
        self.timestamp = datetime.now()
        self.database: Database = Database.DEFAULT.value
        self.crypto_models:List[AbstractOHLCV] = crypto_models
        self.account_holdings = [0 for _ in self.crypto_models]
        self.prediction_handler = PredictionHandler(lstm_sequence_length=100, database=self.database)
        # self.prediction_handler.predict()
    
    def update_state(self, action) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        self.costs_for_action = self.cost_for_action(action)
        done = False
        info = {}
        self.next_state = self.get_current_state()
        print(f'updating state to {self.next_state}')
        return self.next_state, sum(self.costs_for_action), done, info
    
    def reset_state(self) -> npt.NDArray[np.float16]:
        initial_state = self.get_current_state()
        return initial_state
    
    def get_current_state(self):
        self.total_volume = self.calculate_total_volume()
        self.account_holdings = self.get_account_holdings()
        self.new_crypto_data = self.get_new_crypto_data()
        self.usdc_held = self.get_liquidity()
        return np.array([self.total_volume, self.usdc_held] + self.account_holdings + self.new_crypto_data)
    
    def calculate_total_volume(self):
        return sum([float(val) for _, val in self.wallets_dict.items()])
    
    def get_account_holdings(self):
        return [float(self.wallets_dict[crypto_model.symbol]) if crypto_model.symbol in self.wallets_dict.keys() else 0.0 for crypto_model in crypto_models]
    
    def get_liquidity(self) -> float:
        return float(self.wallets_dict['USDC'])
            
    def get_new_crypto_data(self):
        all_entries = []
        for crypto in self.crypto_models:
            crypto_latest = crypto.objects.using(self.database).latest('timestamp')
            if (crypto_latest == None):
                crypto_latest = crypto.default_entry(self.timestamp)
            all_entries = all_entries + self.crypto_to_list(crypto_latest)
            all_entries = all_entries + self.get_new_prediction_data(crypto, crypto_latest.timestamp)
        return all_entries
    
    def get_new_prediction_data(self, crypto_model:AbstractOHLCV, timestamp:datetime) -> List[float]:
        all_entries = []
        ml_models = ['LSTM', 'XGBoost']
        prediction_shifts = [1,24,168]
        for model in ml_models:
            for prediction_shift in prediction_shifts:
                try:
                    entry = Prediction.objects.using(self.database).get(
                        crypto=crypto_model.__name__,
                        timestamp_predicted_for = timestamp, 
                        model_name = model, 
                        predicted_field = f'close_higher_shifted_{prediction_shift}h'
                    )
                    if (np.isnan(entry.predicted_value)):
                        all_entries.append(0.0)
                    else:
                        all_entries.append(entry.predicted_value)
                except Prediction.DoesNotExist:
                    print(f'Prediction {crypto_model.__name__}, {timestamp}, {model}, close_higher_shifted_{prediction_shift}h Does not exist')
                    all_entries.append(0.0)
        return all_entries
    
    def cost_for_action(self, action):
        print('TODO: Implement cost for action method')
        return [0]
    
    def get_crypto_features(self) -> List[str]:
        return crypto_features

    def get_crypto_predicted_features(self) -> List[str]:
        return crypto_predicted_features
    
    def crypto_to_list(self, crypto: AbstractOHLCV) -> List[float]:
        attributes = [getattr(crypto, fieldname)for fieldname in self.get_crypto_features()]
        
        return [attr if attr != None else 0.0 for attr in attributes]
    
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
        cost_for_action = sum(self.cost_for_action(action))
        return reward - cost_for_action*0.25 #! not full scope since this would scale awkwardly. Cost = 4 usdc = -4, which is not reachable the other way around
    
    def check_total_liquidity(self) -> None:
        #! not needed anymore
        crypto_wallets = cb_auth.restClientInstance.get_accounts()
        #! TODO: Implement more fetching if has_next is true
        wallets = crypto_wallets['accounts']
        total_liquidity = 0
        for wallet in wallets:
            currency = wallet["available_balance"]["currency"]
            value = wallet["available_balance"]["value"]
            crypto_model_list = [cpt_model for cpt_model in crypto_models if cpt_model.symbol == currency]
            if (len(crypto_model_list) == 0):
                # can not find: USDC, ETH2, EUR. 
                if currency == 'ETH2': #special case, is called eth2 but in trade is called eth
                    crypto_model = ETH
                    crypto_candles = cb_auth.restClientInstance.get_candles(f'ETH-USDC', int(datetime.timestamp(start)), int(datetime.timestamp(end)), granularity=Granularities.ONE_HOUR.value)
                    price = crypto_candles['candles'][0]['close']
                    final_value = float(value) * float(price)
                    print(f'final value for ETH: {final_value} = {float(value)} * {float(price)}')
                elif currency == 'EUR':
                    crypto_candles = cb_auth.restClientInstance.get_candles(f'USDC-EUR', int(datetime.timestamp(start)), int(datetime.timestamp(end)), granularity=Granularities.ONE_HOUR.value)
                    price = crypto_candles['candles'][0]['close']
                    final_value = float(value) / float(price)
                    print(f'final value for EUR: {final_value}')
                elif currency == 'USDC':
                    final_value = float(value)
                    print(f'final value for USDC: {final_value}')
                else:
                    print(f'could not find {currency} as a model. skipping...')
                    continue
            else:
                crypto_model = crypto_model_list[0]
                end = datetime.now()
                start = end - timedelta(hours=2, minutes=1)
                crypto_candles = cb_auth.restClientInstance.get_candles(f'{crypto_model.symbol}-USDC', int(datetime.timestamp(start)), int(datetime.timestamp(end)), granularity=Granularities.ONE_HOUR.value)
                price = crypto_candles['candles'][-1]['close']
                final_value = float(value) * float(price)
                print(f'final value for {crypto_model.symbol}: {final_value} = {float(value)} * {float(price)}')
            total_liquidity += final_value
        print(f'total liquidity: {total_liquidity}')
    
    
    
    # get_reward_ratios_for_current_timestep()
    
    # setp_count
    
    # initial_volume
    
    # timestamp
    
    # get_liquidity_string()
    
    # cost_for_action()