import asyncio
import requests
import websockets
from coinbase import jwt_generator
from django.core.management.base import BaseCommand
from coinbase_api.constants import SECOND_API_KEY, SECOND_API_SECRET
import time
import json
import jwt
import hashlib
import os
import websocket
import threading
from datetime import datetime, timedelta
from coinbase_api.constants import crypto_models
from typing import Any, Dict
import numpy as np

class Order:
    def __init__(self, order:dict[str: Any]):
        #! needed for inducing bracket sell order
        self.order_side = order["order_side"]
        self.status = order["status"]
        self.avg_price = order["avg_price"]
        self.cumulative_quantity = order["cumulative_quantity"]
        self.total_value_after_fees = order["total_value_after_fees"]
        self.product_id = order["product_id"]
        #! next two are just for sanity check
        self.completion_percentage = order["completion_percentage"]
        self.leaves_quantity = order["leaves_quantity"]
        
    def is_finished_buy_order(self) -> bool:
        # take each condition step by step
        if self.order_side == "BUY":
            print("order is buy side... check!")
            if self.status == "FILLED":
                print("status is FILLED... check!")
                if float(self.completion_percentage) >= 100:
                    print("completion_percentage is 100... check!")
                    if float(self.leaves_quantity) >= 0:
                        print("leaves_quantity is 0... check!")
                        return True
        return False
    
class CoinbaseProduct:
    def __init__(self, api_response: dict[str: Any]):
        self.product_id = self.use_if_exists_else_empty(api_response, "product_id")
        self.price = self.use_if_exists_else_0(api_response, "price")
        self.price_percentage_change_24h = self.use_if_exists_else_0(api_response, "price_percentage_change_24h")
        self.volume_24h = self.use_if_exists_else_0(api_response, "volume_24h")
        self.volume_percentage_change_24h = self.use_if_exists_else_0(api_response, "volume_percentage_change_24h")
        self.base_increment = self.use_if_exists_else_0(api_response, "base_increment")
        self.quote_increment = self.use_if_exists_else_0(api_response, "quote_increment")
        self.quote_min_size = self.use_if_exists_else_0(api_response, "quote_min_size")
        self.quote_max_size = self.use_if_exists_else_0(api_response, "quote_max_size")
        self.base_min_size = self.use_if_exists_else_0(api_response, "base_min_size")
        self.base_max_size = self.use_if_exists_else_0(api_response, "base_max_size")
        self.base_name = self.use_if_exists_else_empty(api_response, "base_name")
        self.quote_name = self.use_if_exists_else_empty(api_response, "quote_name")
        self.watched = self.use_if_exists_else_false(api_response, "watched")
        self.is_disabled = self.use_if_exists_else_false(api_response, "is_disabled")
        self.new = self.use_if_exists_else_false(api_response, "new")
        self.status = self.use_if_exists_else_empty(api_response, "status")
        self.cancel_only = self.use_if_exists_else_false(api_response, "cancel_only")
        self.limit_only = self.use_if_exists_else_false(api_response, "limit_only")
        self.post_only = self.use_if_exists_else_false(api_response, "post_only")
        self.trading_disabled = self.use_if_exists_else_false(api_response, "trading_disabled")
        self.auction_mode = self.use_if_exists_else_false(api_response, "auction_mode")
        self.product_type = self.use_if_exists_else_empty(api_response, "product_type")
        self.quote_currency_id = self.use_if_exists_else_0(api_response, "quote_currency_id")
        self.base_currency_id = self.use_if_exists_else_0(api_response, "base_currency_id")
        self.fcm_trading_session_details = self.use_if_exists_else_empty(api_response, "fcm_trading_session_details")
        self.mid_market_price = self.use_if_exists_else_0(api_response, "mid_market_price")
        self.alias = self.use_if_exists_else_empty(api_response, "alias")
        self.alias_to = self.use_if_exists_else_empty(api_response, "alias_to")
        self.base_display_symbol = self.use_if_exists_else_empty(api_response, "base_display_symbol")
        self.quote_display_symbol = self.use_if_exists_else_empty(api_response, "quote_display_symbol")
        self.view_only = self.use_if_exists_else_false(api_response, "view_only")
        self.price_increment = self.use_if_exists_else_0(api_response, "price_increment")
        self.display_name = self.use_if_exists_else_empty(api_response, "display_name")
        self.product_venue = self.use_if_exists_else_empty(api_response, "product_venue")
        self.approximate_quote_24h_volume = self.use_if_exists_else_0(api_response, "approximate_quote_24h_volume")
        self.new_at = self.use_if_exists_else_empty(api_response, "new_at")
    
    def use_if_exists_else_0(self, api_response: dict[str: Any], key: str) -> float:
        if key in api_response.keys():
            try:
                return float(api_response[key])
            except:
                return 0.0
        return 0.0
    
    def use_if_exists_else_empty(self, api_response: dict[str: Any], key: str) -> str:
        if key in api_response.keys():
            return api_response[key]
        return ""
    
    def use_if_exists_else_false(self, api_response: dict[str: Any], key: str) -> bool:
        if key in api_response.keys():
            try:
                return bool(api_response[key])
            except:
                return False
        return False
    
    def __str__(self):
        return f"{self.product_id}:{self.base_increment}:{self.quote_increment}"
    
def load_product_data() -> dict[str: CoinbaseProduct]:
    filename = "product_data.json"
    with open(filename) as f:
        d = json.load(f)
        print(f"loaded json from file {filename}")
        # print(d)
    
    ret_dict = {
        product["product_id"]:CoinbaseProduct(product) for product in d["products"]
    }
    # print(f"ret dict: {ret_dict}")
    
    return ret_dict

def get_base_increment(coinbase_product: CoinbaseProduct):
    print(f"base_increment: {coinbase_product.base_increment}")
    print(f"base_increment: {np.format_float_positional(coinbase_product.base_increment)}")
    return len(np.format_float_positional(coinbase_product.base_increment).split(".")[1])

def get_quote_increment(coinbase_product: CoinbaseProduct):
    print(f"quote_increment: {coinbase_product.quote_increment}")
    return len(np.format_float_positional(coinbase_product.quote_increment).split(".")[1])

def get_price_increment(coinbase_product: CoinbaseProduct):
    print(f"price_increment: {coinbase_product.price_increment}")
    return len(np.format_float_positional(coinbase_product.price_increment).split(".")[1])

def get_product_data(product_id: str) -> CoinbaseProduct:
    product_data = load_product_data()
    if product_id in product_data:
        return product_data[product_id]
    else:
        return Exception(f"product id {product_id} not found in product data")
    
def place_sell_order(buy_order: Order):
    if (not buy_order.is_finished_buy_order()):
        print(f"buy order is not finished order!")
        return
    print(f"Attempting to sell {buy_order.product_id}.")
    product_data: CoinbaseProduct = get_product_data(buy_order.product_id)
    side = "SELL"
    base_price = buy_order.avg_price
    lower_price = float(base_price) * 0.99
    upper_price = float(base_price) * 1.03
    product_id = buy_order.product_id
    request_method = "POST"
    request_path = "/api/v3/brokerage/orders/preview"
    # request_path = "/api/v3/brokerage/orders"
    base_url = "https://api.coinbase.com"
    jwt_uri = jwt_generator.format_jwt_uri(request_method, request_path)
    jwt_token = jwt_generator.build_rest_jwt(jwt_uri, SECOND_API_KEY, SECOND_API_SECRET)
    # print(jwt_token)
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    base_size = float(buy_order.cumulative_quantity)
    base_increment = get_base_increment(product_data)
    price_increment = get_price_increment(product_data)
    print(f"unformatted: base_size: {base_size}. upper_price: {upper_price}. lower_price: {lower_price}")
    print(f"formatted: base_size: {base_size:.{base_increment}f}. upper_price: {upper_price:.{price_increment}f}. lower_price: {lower_price:.{price_increment}f}")
    payload = {
        "product_id": product_id,
        "side": side,
        "order_configuration": {
            "trigger_bracket_gtc": {
                "base_size": f"{base_size:.{base_increment}f}",
                "limit_price": f"{upper_price:.{price_increment}f}",
                "stop_trigger_price": f"{lower_price:.{price_increment}f}"
            }
        }
    }
    response = requests.post(f"{base_url}{request_path}", headers=headers, json=payload)
    response_decoded = json.loads(response.content.decode())
    print(f"response_decoded:\n{response_decoded}")
    

ALGORITHM = "ES256"

CHANNEL_NAMES = {
    "level2": "level2",
    "user": "user",
    "tickers": "ticker",
    "ticker_batch": "ticker_batch",
    "status": "status",
    "market_trades": "market_trades",
    "candles": "candles",
}

if not SECOND_API_SECRET or not SECOND_API_KEY:
    raise ValueError("Missing mandatory environment variable(s)")

WS_API_URL = "wss://advanced-trade-ws-user.coinbase.com"
# WS_API_URL = "wss://advanced-trade-ws.coinbase.com"

# async def client(websocket_url):
#     async for websocket in websockets.connect(websocket_url):
#         print("Connected to Websocket server")
        
#         try:
#             async for message in websocket:
#             # Process message received on the connection.
#                 print(message)
#         except websockets.ConnectionClosed:
#             print("Connection lost! Retrying..")
#             continue #continue will retry websocket connection by exponential back off 
        
def sign_with_jwt(message, channel, products=[]):
    payload = {
        "iss": "coinbase-cloud",
        "nbf": int(time.time()),
        "exp": int(time.time()) + 120,
        "sub": SECOND_API_KEY,
    }
    headers = {
        "kid": SECOND_API_KEY,
        "nonce": hashlib.sha256(os.urandom(16)).hexdigest()
    }
    token = jwt.encode(payload, SECOND_API_SECRET, algorithm=ALGORITHM, headers=headers)
    message['jwt'] = token
    return message

def on_message(ws, message):
    data = json.loads(message)
    events = data["events"]
    num_events = len(events)
    for j in range(num_events):
        orders = events[j]["orders"]
        num_orders = len(events)
        for i in range(num_orders):
            new_order = Order(orders[i])
            # avg_price = orders[i]["avg_price"]
            # completion_percentage = orders[i]["completion_percentage"]
            # cumulative_quantity = orders[i]["cumulative_quantity"]
            # filled_value = orders[i]["filled_value"]
            # leaves_quantity = orders[i]["leaves_quantity"]
            # limit_price = orders[i]["limit_price"]
            # order_side = orders[i]["order_side"]
            # product_id = orders[i]["product_id"]
            # status = orders[i]["status"]
            # total_value_after_fees = orders[i]["total_value_after_fees"]
            print(f"""{new_order.product_id}. {new_order.order_side}. {new_order.status}:
                \tavg_price:\t\t{new_order.avg_price}
                \tcompletion_percentage:\t{new_order.completion_percentage}
                \tcumulative_quantity:\t{new_order.cumulative_quantity}
                \tleaves_quantity:\t{new_order.leaves_quantity}
                \ttotal_value_after_fees:\t{new_order.total_value_after_fees}
                """)
            place_sell_order(new_order)
    with open("Output1.txt", "a") as f:
        f.write(json.dumps(data) + "\n")

def subscribe_to_products(ws, products, channel_name):
    message = {
        "type": "subscribe",
        "channel": channel_name,
        "product_ids": products
    }
    signed_message = sign_with_jwt(message, channel_name, products)
    ws.send(json.dumps(signed_message))

def unsubscribe_to_products(ws, products, channel_name):
    message = {
        "type": "unsubscribe",
        "channel": channel_name,
        "product_ids": products
    }
    signed_message = sign_with_jwt(message, channel_name, products)
    ws.send(json.dumps(signed_message))

def on_open(ws):
    products = get_products()
    subscribe_to_products(ws, products, CHANNEL_NAMES["user"])
    # subscribe_to_products(ws, products, CHANNEL_NAMES["level2"])

def start_websocket():
    ws = websocket.WebSocketApp(WS_API_URL, on_open=on_open, on_message=on_message, on_error=print)
    ws.run_forever()
    
def get_products():
    return [f"{crypto.symbol}-USDC" for crypto in crypto_models]
    
    

class Command(BaseCommand):

    def handle(self, *args, **options):
        # jwt_token = jwt_generator.build_ws_jwt(SECOND_API_KEY, SECOND_API_SECRET)
        # URL = "wss://advanced-trade-ws-user.coinbase.com"
        # print(f"Connecting to websocket server {URL}")
        # asyncio.run(client(URL))
        #! temporarily testing preview order
        order_data = {
            "order_side": "BUY",
            "status": "FILLED",
            "avg_price": "3.140",
            "cumulative_quantity": "0.1",
            "total_value_after_fees": "0.3470103",
            "product_id": "BTC-USDC",
            # "product_id": "NEAR-USDC",
            "completion_percentage": "100.00",
            "leaves_quantity": "0",
        }
        # self.order_side = order["order_side"]
        # self.status = order["status"]
        # self.avg_price = order["avg_price"]
        # self.cumulative_quantity = order["cumulative_quantity"]
        # self.total_value_after_fees = order["total_value_after_fees"]
        # self.product_id = order["product_id"]
        # #! next two are just for sanity check
        # self.completion_percentage = order["completion_percentage"]
        # self.leaves_quantity = order["leaves_quantity"]
        order:Order = Order(order_data)
        place_sell_order(order)
        return
        ws_thread = threading.Thread(target=start_websocket)
        ws_thread.start()

        sent_unsub = False
        start_time = datetime.utcnow()

        try:
            # while True:
                if (datetime.utcnow() - start_time).total_seconds() > 5 and not sent_unsub:
                    # Unsubscribe after 5 seconds
                    ws = websocket.create_connection(WS_API_URL)
                    products = get_products()
                    unsubscribe_to_products(ws, products, CHANNEL_NAMES["user"])
                    ws.close()
                    sent_unsub = True
                time.sleep(1)
        except Exception as e:
            print(f"Exception: {e}")