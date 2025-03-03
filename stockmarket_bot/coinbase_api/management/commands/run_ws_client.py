import uuid
from django.core.management.base import BaseCommand
from coinbase_api.classes.CoinbaseProduct import CoinbaseProduct
from coinbase_api.classes.Order import Order
from coinbase_api.constants import SECOND_API_KEY, SECOND_API_SECRET, ALGORITHM, CHANNEL_NAMES, WS_API_URL
import time
import json
import jwt
import hashlib
import os
import websocket
import threading
from datetime import datetime
from coinbase_api.constants import crypto_models
import numpy as np

from coinbase_api.enums import ApiPath, Method, Side
from coinbase_api.utilities.utils import api_request_with_auth

    
def load_product_data() -> dict[str: CoinbaseProduct]:
    filename = "product_data.json"
    with open(filename) as f:
        d = json.load(f)
        print(f"loaded product data json from file {filename}")
    return {
        product["product_id"]:CoinbaseProduct(product) for product in d["products"]
    }

def get_base_increment(coinbase_product: CoinbaseProduct):
    return len(np.format_float_positional(coinbase_product.base_increment).split(".")[1])

def get_quote_increment(coinbase_product: CoinbaseProduct):
    return len(np.format_float_positional(coinbase_product.quote_increment).split(".")[1])

def get_price_increment(coinbase_product: CoinbaseProduct):
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
    base_price = buy_order.avg_price
    lower_price = float(base_price) * 0.99
    upper_price = float(base_price) * 1.03
    base_size = float(buy_order.cumulative_quantity)
    base_increment = get_base_increment(product_data)
    price_increment = get_price_increment(product_data)
    payload = {
        "client_order_id": str(uuid.uuid4()),
        "product_id": buy_order.product_id,
        "side": Side.SELL.value,
        "order_configuration": {
            "trigger_bracket_gtc": {
                "base_size": f"{base_size:.{base_increment}f}",
                "limit_price": f"{upper_price:.{price_increment}f}",
                "stop_trigger_price": f"{lower_price:.{price_increment}f}"
            }
        }
    }
    response_decoded = api_request_with_auth(ApiPath.ORDERS.value, Method.POST, request_body = payload)
    print(f"response_decoded:\n{response_decoded}")

if not SECOND_API_SECRET or not SECOND_API_KEY:
    raise ValueError("Missing mandatory environment variable(s)")
     
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

def start_websocket():
    ws = websocket.WebSocketApp(WS_API_URL, on_open=on_open, on_message=on_message, on_error=print)
    ws.run_forever()
    
def get_products():
    return [f"{crypto.symbol}-USDC" for crypto in crypto_models]
    

class Command(BaseCommand):

    def handle(self, *args, **options):
        # order_data = {
        #     "order_side": "BUY",
        #     "status": "FILLED",
        #     "avg_price": "3.140",
        #     "cumulative_quantity": "0.1",
        #     "total_value_after_fees": "0.3470103",
        #     "product_id": "NEAR-USDC",
        #     "completion_percentage": "100.00",
        #     "leaves_quantity": "0",
        # }
        # order:Order = Order(order_data)
        # place_sell_order(order)
        # return
        ws_thread = threading.Thread(target=start_websocket)
        ws_thread.start()

        sent_unsub = False
        start_time = datetime.utcnow()

        try:
            while True:
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