from datetime import datetime, timedelta, timezone
import json
import logging
import uuid

import numpy as np

from coinbase_api.classes.CoinbaseProduct import CoinbaseProduct
from coinbase_api.enums import ApiPath, Method, Side
from coinbase_api.models.models import AbstractOHLCV
from coinbase_api.models.singleton_meta import SingletonMeta
from coinbase_api.utilities.utils import api_request_with_auth
from coinbase_api.constants import crypto_models

from coinbase_api.utilities.cb_provider import CbProvider
cb_provider = CbProvider()

class RlActionHandler(metaclass=SingletonMeta):
    # """
    # Singleton class for Coinbase data handling.
    # """
    # _instance = None
    
    # def __new__(cls):
    #     """
    #     Override the __new__ method to control the object creation process.
    #     :return: A single instance of CBAuth
    #     """
    #     if cls._instance is None:
    #         print("Creating CBProvider instance")
    #         cls._instance = super(RlActionHandler, cls).__new__(cls)
    #         cls._instance.init()
    #     return cls._instance

    def __init__(self):
        """
        Initialize the CBAuth instance with API credentials.
        """
        logging.basicConfig(
            filename='rl_action_handler.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s',
            filemode='a'
        )
        self.product_data = {}
        self.logger = logging.getLogger(__name__)
        self.product_data = self.load_product_data()
        
    def handle_actions(self, action: dict[AbstractOHLCV, float]):
        #! first decide upon which actions to actually use
        sorted_actions = {k: v for k, v in sorted(action.items(), key=lambda item: item[1])}
        keys_to_perform_action:list[AbstractOHLCV] = list(sorted_actions.keys())[-3:]
        self.wallets_dict = cb_provider.get_wallets()
        current_liquidity = cb_provider.get_liquidity()
        if (current_liquidity < 5.0):
            self.logger.info(f"Current liquidity is below 5$. Not performing any actions!")
            return
        for key in keys_to_perform_action:
            quote_size = min(100, float(current_liquidity)/3.0)
            print(f'act: {sorted_actions[key]} on crypto {key.symbol}')
            _, quote_increment, price_increment = self.get_increments()
            best_bid = self.get_crypto_data_price(key)
            end_time = datetime.now(timezone.utc) + timedelta(minutes=15) #! valid for 15 minutes only
            payload = {
                # "client_order_id": str(uuid.uuid4()),
                "product_id": f"{key.symbol}-USDC",
                "side": Side.BUY.value,
                "order_configuration": {
                    "limit_limit_gtd": {
                        "quote_size": f"{quote_size:.{quote_increment}f}",
                        "limit_price": f"{best_bid:.{price_increment}f}",
                        "end_time": end_time.isoformat()
                    }
                }
            }
            result = self.place_buy_order(payload=payload)
            print(f'preview: {result}')
            break
        
    def get_crypto_data_price(self, crypto: AbstractOHLCV):
        raw_data = api_request_with_auth(
            request_path=ApiPath.BEST_BID_ASKS.value,
            request_method=Method.GET,
            request_body = {
                "product_candles": f"{crypto.symbol}-USDC"
                }
            )
        pricebooks = raw_data["pricebooks"]
        try:
            best_bid = float(pricebooks["bids"])
        except Exception as e:
            print(f'Encountered the following error: {e}')
        return best_bid 
        
    def place_buy_order(self, payload = {}):
        response_decoded = api_request_with_auth(ApiPath.ORDERS_PREVIEW.value, Method.POST, request_body = payload)
        # response_decoded = api_request_with_auth(ApiPath.ORDERS.value, Method.POST, request_body = payload)
        return response_decoded
        
    def get_increments(self, crypto: AbstractOHLCV):
        product_data: CoinbaseProduct = self.get_product_data(f"{crypto.symbol}-USDC")
        base_increment = self.get_base_increment(product_data)
        quote_increment = self.get_quote_increment(product_data)
        price_increment = self.get_price_increment(product_data)
        return base_increment, quote_increment, price_increment
        
    def load_product_data(self) -> dict[str: CoinbaseProduct]:
        if self.product_data:
            return self.product_data
        
        filename = "product_data.json"
        with open(filename) as f:
            d = json.load(f)
            print(f"loaded product data json from file {filename}")
        product_data = {
            product["product_id"]:CoinbaseProduct(product) for product in d["products"]
        }
        all_crypto_product_ids = [f"{crypto.symbol}-USDC" for crypto in crypto_models]
        self.product_data = {k:v for k,v in product_data.items() if k in all_crypto_product_ids}
        return self.product_data
        
    def get_product_data(self, product_id: str) -> CoinbaseProduct:
        product_data = self.load_product_data()
        if product_id in product_data:
            return product_data[product_id]
        else:
            return Exception(f"product id {product_id} not found in product data")
        
    def get_base_increment(self, coinbase_product: CoinbaseProduct):
        return len(np.format_float_positional(coinbase_product.base_increment).split(".")[1])

    def get_quote_increment(self, coinbase_product: CoinbaseProduct):
        return len(np.format_float_positional(coinbase_product.quote_increment).split(".")[1])

    def get_price_increment(self, coinbase_product: CoinbaseProduct):
        return len(np.format_float_positional(coinbase_product.price_increment).split(".")[1])
