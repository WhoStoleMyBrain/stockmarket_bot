# base_app/management/commands/setup_periodic_task.py
from datetime import datetime, timedelta, timezone
from django.core.management.base import BaseCommand, CommandError
# from celery_app.tasks import print_statement
from coinbase_api.constants import crypto_models, crypto_extra_features, crypto_features, crypto_predicted_features
import os
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from gymnasium import spaces
from coinbase_api.enums import Database
from coinbase_api.ml_models.RL_decider_model import CustomEnv
from coinbase_api.ml_models.data_handlers.real_data_handler import RealDataHandler
import numpy as np
from coinbase_api.models.models import Account
from coinbase_api.views.views import cb_auth

class Command(BaseCommand):
    help = 'Test the real environment WITHOUT actually modifying the state'

    def handle(self, *args, **kwargs):
        data_handler = RealDataHandler()
        # action = [0.0 for i in range(len(crypto_models))]
        model_path = 'coinbase_api/ml_models/rl_model.pkl'
        # data_handler.update_state(action)
        if os.path.exists(model_path):
            # Load the existing model
            print('Loading model!')
            env = CustomEnv(data_handler=data_handler)
            model = PPO.load(model_path, env=env)
            print('Loaded model!')
        else:
            raise NotImplementedError(f"PPO Model on path {model_path} does not exist. Real data application not available!")
        vec_env = model.get_env()
        obs = vec_env.reset()
        # print(f'observation before step: {obs}. {len(obs)}')
        for idx, entry in enumerate(obs[-1]):
            if (np.isnan(entry)):
                print(f'{idx}: {entry}')
        action, states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # obs, reward, done, info = model.env.step(obs)
        # print(f'observation after step: {obs}. {len(obs)}')
        action, _ = model.predict(obs, deterministic=False)
        # print(f'Action trying to take: {action}')
        self.handle_actions(action, model, obs)
        
    def handle_actions(self, action, model:PPO, obs):
        for idx, act in enumerate(action[0]):
            crypto = crypto_models[idx]
            account = self.get_crypto_account(crypto.symbol)
            print(f'act: {act} on crypto {crypto.symbol}. account: {account.name}:{account.currency}')
            # limit order buy FOK -> Fill or Kill -> either full order is filled or it is canceled
            # limit order buy IOC -> Immediate or Cancel -> Buy immediately or cancel any unfulfilled
            # limit order buy GTD -> Good till Date -> Zeitliche Einschränkung
            # limit order buy GTC -> Good till Canceled -> Bleibt bis manuell geschlossen
            endtime = datetime.now(timezone.utc) + timedelta(minutes=5)
            result = cb_auth.restClientInstance.preview_limit_order_gtd_buy(product_id=f'{crypto.symbol}-USDC', base_size="10.0", limit_price="10000.0", end_time=endtime.isoformat()) # use base size for base value (i.e. BTC) and quote_size for USDC
            # result = cb_auth.restClientInstance.preview_market_order_buy(product_id=f'{crypto.symbol}-USDC', quote_size="10.0") # use base size for base value (i.e. BTC) and quote_size for USDC
            print(f'preview: {result}')
            
            break
            
    def get_crypto_account(self, symbol: str) -> Account:
        try:
            crypto_account = Account.objects.using(Database.DEFAULT.value).get(name=f'{symbol} Wallet')
            return crypto_account
        except Account.DoesNotExist:
            print(f'Account {symbol} Wallet does not exist!')
            raise Account.DoesNotExist