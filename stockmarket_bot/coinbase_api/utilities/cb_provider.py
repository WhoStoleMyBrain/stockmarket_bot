from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
from typing import List

import numpy as np
import pandas as pd
import ta
from coinbase_api.constants import SECOND_API_KEY, SECOND_API_SECRET, crypto_models, crypto_features
from coinbase_api.enums import ApiPath, Granularities, Method
from coinbase_api.models.models import AbstractOHLCV
from coinbase_api.models.singleton_meta import SingletonMeta
from coinbase_api.utilities.utils import api_request_with_auth



class CbProvider(metaclass=SingletonMeta):

    def __init__(self):
        """
        Initialize the CBAuth instance with API credentials.
        """
        logging.basicConfig(
            filename='cb_provider.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s',
            filemode='a'
        )
        self.logger = logging.getLogger(__name__)
        self.key = SECOND_API_KEY
        self.secret = SECOND_API_SECRET
        self.update()
        
    def set_credentials(self, api_key, api_secret):
        """
        Update the API credentials used for authentication.
        :param api_key: The API Key for Coinbase API
        :param api_secret: The API Secret for Coinbase API
        """
        self.key = api_key
        self.secret = api_secret
        
    def update(self):
        self.update_wallets()
        self.update_prices()
        self.update_crypto_data()
        
    def update_wallets(self):
        crypto_wallets = api_request_with_auth(ApiPath.ACCOUNTS.value, Method.GET)
        self.wallets_dict = {wallet["available_balance"]["currency"]: wallet["available_balance"]["value"] for wallet in crypto_wallets["accounts"]}
        
    def update_crypto_data(self):
        self.data_dict = {}
        #! hold candles for all cryptos that are being used in dict.
        for crypto in crypto_models:
            all_entries = []
            start = datetime.now() - timedelta(minutes=5*50)
            end = datetime.now() + timedelta(minutes=1)
            raw_data = self.get_crypto_raw_candles(start, end, crypto.symbol)
            crypto_instances = self.parse_raw_data(crypto, raw_data["candles"])
            crypto_instance_now: AbstractOHLCV = self.add_calculated_parameters(crypto_instances)
            all_entries = all_entries + self.crypto_to_list(crypto_instance_now)
            self.data_dict[crypto.symbol] = all_entries
            
    def update_prices(self):
        request_body = {
            "get_all_products": "true",
        }
        all_products = api_request_with_auth(ApiPath.PRODUCTS.value, request_method=Method.GET, request_body=request_body)
        products = all_products["products"]
        usable_products = [item for item in products if item["product_id"] in self.get_product_types()]
        self.prices_dict = {item["product_id"]: item["price"] for item in usable_products}
    
        
    def get_wallets(self) -> dict:
        return self.wallets_dict
    
    def get_account_holdings(self):
        return [float(self.wallets_dict[crypto_model.symbol]) if crypto_model.symbol in self.wallets_dict.keys() else 0.0 for crypto_model in crypto_models]
    
    def get_account_holding(self, crypto: AbstractOHLCV):
        return float(self.wallets_dict[crypto.symbol]) if crypto.symbol in self.wallets_dict.keys() else 0.0
    
    def get_liquidity(self):
        if ("USDC" in self.wallets_dict):
            return float(self.wallets_dict["USDC"])
        self.logger.error(f"Did not find USDC in wallet dict. No actions can be taken!")
        return 0.0
    
    def get_product_types(self):
        return [f"{crypto.symbol}-USDC" for crypto in crypto_models]
    
    def get_prices(self):
        return self.prices_dict
    
    def get_total_volume(self):
        total = sum([float(val) * float(self.prices_dict[f"{key}-USDC"]) for key, val in self.wallets_dict.items() if f"{key}-USDC" in self.prices_dict.keys()])
        total_with_usdc = total + self.get_liquidity()
        return total_with_usdc
    
    def get_crypto_data(self, crypto_model: AbstractOHLCV):
        return self.data_dict[crypto_model.symbol]
    
      
    def parse_raw_data(self, crypto_model: AbstractOHLCV, data) -> List[AbstractOHLCV]:
        try:
            entries = []
            for item in data:
                try:
                    timestamp = float(item["start"])
                    low = float(item["low"])
                    high = float(item["high"])
                    opening = float(item["open"])
                    close_base = float(item["close"])
                    volume = float(item["volume"])
                    
                    new_entry = crypto_model(
                        timestamp=datetime.fromtimestamp(int(timestamp), tz=timezone.utc),
                        open=opening,
                        high=high,
                        low=low,
                        close=close_base,
                        volume=volume,
                    )
                    entries.append(new_entry)
                except Exception as e:
                    print(f'Encountered an error with item {item}: {e}')
            return entries
        except Exception as e:
            print(f'Encountered the following error: {e}')

    def get_crypto_raw_candles(self, start, end, crypto_symbol):
        raw_data = api_request_with_auth(
            request_path=f"{ApiPath.PRODUCTS.value}/{crypto_symbol}-USDC/candles",
            request_method=Method.GET,
            request_body = {
                "start" : str(int(datetime.timestamp(start))),
                "end": str(int(datetime.timestamp(end))),
                "granularity": Granularities.FIVE_MINUTE.value
            })
        return raw_data
    
    def get_crypto_features(self) -> List[str]:
        return crypto_features
    
    def crypto_to_list(self, crypto: AbstractOHLCV) -> List[float]:
        attributes = [getattr(crypto, fieldname)for fieldname in self.get_crypto_features()]
        return [attr if attr != None else 0.0 for attr in attributes]
    
    def calculate_vmap(self, data: List[AbstractOHLCV]):
        total_volume = 0.0
        total_value = 0.0
        for item in data:
            # Ensure that volume is available; skip items with no volume
            if item.volume is None:
                continue
            volume = item.volume
            # Use 0 as a fallback if any price is missing (you may adjust this logic)
            open_price = item.open if item.open is not None else 0.0
            close_price = item.close if item.close is not None else 0.0
            high_price = item.high if item.high is not None else 0.0
            # Compute the average price for this item
            price = (open_price + close_price + high_price) / 3
            total_volume += volume
            total_value += price * volume

            return total_value / total_volume if total_volume != 0 else 0

    def calculate_percentage_returns(self, current_item:AbstractOHLCV, previous_item:AbstractOHLCV):
        if previous_item is None:
            return None
        if previous_item.close == 0:
            return 0.0
        else:
            return (current_item.close - previous_item.close) / (previous_item.close)
        
    def calculate_log_returns(self, current_item:AbstractOHLCV, previous_item:AbstractOHLCV):
        if previous_item is None:
            return None
        if abs(current_item.close - previous_item.close) < 1e-15:
            return 0.0
        else:
            return np.log(current_item.close / previous_item.close)
        
    def check_nan(self, value):
        return value if not pd.isna(value) else None
    
    def add_calculated_parameters(self, all_data: list[AbstractOHLCV]) -> AbstractOHLCV:
        sequence_length = 50
        rsi_length = 14
        bollinger_length = 20
        # batch_size = 500  #! batch size not needed since we will handle only single data
        # -------------------------------------------------------------------------
        # 1. Preload all data into memory, sorted by timestamp (ascending)
        #    This avoids multiple DB hits when iterating.
        # -------------------------------------------------------------------------
        # all_data = list(crypto_model.objects.using(database).order_by('timestamp').all())
        total_length = len(all_data)
        if total_length < sequence_length:
            print(f"Did not have enough data for timestamp: {all_data[-1].timestamp}")
            return all_data[-1]
        
        day_data = defaultdict(list)
        for item in all_data:
            day_data[item.timestamp.date()].append(item)
        # Sort each day’s items in descending order to mimic order_by('-timestamp')
        for day, items in day_data.items():
            items.sort(key=lambda x: x.timestamp, reverse=True)
        
        close_values = [item.close for item in all_data]


        previous_item = all_data[-2]
        idx = total_length - 1
        item = all_data[idx]

        # Skip items that are already updated
        if item.all_fields_set():
            return item

        # Retrieve data for the day using the precomputed grouping
        day = item.timestamp.date()
        data_day_of_item = day_data.get(day, [])
        
        # Obtain the sliding window of closing prices
        start_idx = max(0, idx - sequence_length + 1)
        sequence_closes = close_values[start_idx:]
        close_series = pd.Series(sequence_closes)
        
        # close = pd.Series(close)
        close_bollinger = close_series[-bollinger_length:]
        close_rsi = close_series[-rsi_length:]

        # Calculate parameters
        item.vmap = self.calculate_vmap(data_day_of_item)
        item.percentage_returns = self.calculate_percentage_returns(item, previous_item)
        item.log_returns = self.calculate_log_returns(item, previous_item)
        item.sma = self.check_nan(ta.trend.sma_indicator(close_series, sequence_length).iloc[-1])
        item.ema = self.check_nan(ta.trend.ema_indicator(close_series, sequence_length).iloc[-1])
        item.macd = self.check_nan(
            ta.trend.ema_indicator(close_series, 12).iloc[-1] - ta.trend.ema_indicator(close_series, 26).iloc[-1]
        )
        bollinger = ta.volatility.BollingerBands(close_bollinger, bollinger_length, 2)
        item.bollinger_high = self.check_nan(bollinger.bollinger_hband().iloc[-1])
        item.bollinger_low = self.check_nan(bollinger.bollinger_lband().iloc[-1])
        item.rsi = self.check_nan(ta.momentum.rsi(close_rsi, rsi_length).iloc[-1])

        return item
        
        