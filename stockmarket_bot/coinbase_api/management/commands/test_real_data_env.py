# base_app/management/commands/setup_periodic_task.py
import json
import uuid
from django.core.management.base import BaseCommand
# from celery_app.tasks import print_statement
from coinbase_api.enums import ApiPath, Method, Side
from coinbase_api.models.models import AbstractOHLCV
from coinbase_api.utilities.cb_provider import CbProvider
from coinbase_api.utilities.rl_action_handler import RlActionHandler
from coinbase_api.utilities.rl_provider import RlProvider
from coinbase_api.utilities.utils import api_request_with_auth

cb_provider = CbProvider()
rl_provider = RlProvider()
rl_action_handler = RlActionHandler()
class Command(BaseCommand):
    help = 'Test the real environment WITHOUT actually modifying the state'

    def handle(self, *args, **kwargs):
        # self.crypto = BTC
        # all_actions = {}
        # data_handler = RealDataHandler(self.crypto)
        # model_path = 'coinbase_api/ml_models/rl_model.pkl'
        # if os.path.exists(model_path):
        #     # Load the existing model
        #     print('Loading model!')
        #     self.env = CustomEnv(data_handler=data_handler)
        #     model = PPO.load(model_path, env=self.env)
        #     print('Loaded model!')
        # else:
        #     raise NotImplementedError(f"PPO Model on path {model_path} does not exist. Real data application not available!")
        # vec_env = model.get_env()
        # obs = vec_env.reset()
        # for idx, entry in enumerate(obs[-1]):
        #     if (np.isnan(entry)):
        #         print(f'{idx}: {entry}')
        # #! iterate over crypto models
        # for crypto_model in crypto_models:
        #     action, states = model.predict(obs, deterministic=True)
        #     self.env.set_currency(crypto_model)
        #     model.set_env(self.env)
        #     obs, reward, done, info = vec_env.step(action)
        #     action, _ = model.predict(obs, deterministic=False)
        #     all_actions[crypto_model] = action[0] # collect actions
        cb_provider.update()
        all_actions = rl_provider.get_all_actions()
        rl_action_handler.handle_actions(all_actions)
        
    def handle_actions(self, action:dict[AbstractOHLCV, float]):
        #! first decide upon which actions to actually use
        sorted_actions = {k: v for k, v in sorted(action.items(), key=lambda item: item[1])}
        keys_to_perform_action:list[AbstractOHLCV] = sorted_actions.keys()[-3:]
        crypto_wallets = api_request_with_auth(ApiPath.ACCOUNTS.value, Method.GET)
        self.wallets_dict = {wallet["available_balance"]["currency"]: wallet["available_balance"]["value"] for wallet in crypto_wallets["accounts"]}
        current_liquidity = self.wallets_dict["USDC"]
        for key in keys_to_perform_action:
            quote_size = min(100, float(current_liquidity)/3)
            print(f'act: {sorted_actions[key]} on crypto {key.symbol}')
            _, quote_increment, price_increment = self.get_increments()
            best_bid = self.get_crypto_data_price(key)
            payload = {
                # "client_order_id": str(uuid.uuid4()),
                "product_id": f"{key.symbol}-USDC",
                "side": Side.BUY.value,
                "order_configuration": {
                    "limit_limit_gtc": {
                        "quote_size": f"{quote_size:.{quote_increment}f}",
                        "limit_price": f"{best_bid:.{price_increment}f}"
                    }
                }
            }
            result = self.place_buy_order(payload=payload)
            print(f'preview: {result}')
            break
        
    