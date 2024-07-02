from typing import Any, Dict, List, Tuple
import numpy as np
import numpy.typing as npt
from coinbase_api.enums import Database, Granularities
from coinbase_api.ml_models.data_handlers.abstract_data_handler import AbstractDataHandler
from coinbase_api.models.generated_models import ETH
from coinbase_api.models.models import AbstractOHLCV, Account
from datetime import datetime, timedelta
from coinbase_api.utilities.prediction_handler import PredictionHandler
from coinbase_api.views.views import cb_auth
from coinbase_api.constants import crypto_models


class RealDataHandler(AbstractDataHandler):
    def __init__(self, initial_volume = 0) -> None:
        crypto_wallets = cb_auth.restClientInstance.get_accounts()
        #! TODO: Implement more fetching if has_next is true
        wallets_dict = {wallet["available_balance"]["currency"]: wallet["available_balance"]["value"] for wallet in crypto_wallets["accounts"]}
        print(wallets_dict)
        self.action_factor = 0.2
        self.database: Database = Database.DEFAULT.value
        self.crypto_models:List[AbstractOHLCV] = crypto_models
        self.account_holdings = [0 for _ in self.crypto_models]
        self.prediction_handler = PredictionHandler(lstm_sequence_length=100, database=self.database, timestamp=self.timestamp)
        pass
    
    def update_state(self, action) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        pass
    
    def reset_state(self) -> npt.NDArray[np.float16]:
        pass
    
    def check_total_liquidity(self) -> None:
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