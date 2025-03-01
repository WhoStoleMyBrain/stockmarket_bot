import logging
import os

from stable_baselines3 import PPO

from coinbase_api.ml_models.RL_decider_model import CustomEnv
from coinbase_api.ml_models.data_handlers.real_data_handler import RealDataHandler
from coinbase_api.constants import crypto_models
from coinbase_api.models.models import AbstractOHLCV
from coinbase_api.models.singleton_meta import SingletonMeta

class RlProvider(metaclass=SingletonMeta):
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
    #         cls._instance = super(RlProvider, cls).__new__(cls)
    #         cls._instance.init()
    #     return cls._instance

    def __init__(self):
        """
        Initialize the CBAuth instance with API credentials.
        """
        logging.basicConfig(
            filename='rl_provider.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s',
            filemode='a'
        )
        self.logger = logging.getLogger(__name__)
        self.crypto_environments = {
            crypto.symbol: RealDataHandler(crypto) for crypto in crypto_models
        }
        self.crypto_models = {}
        model_path = 'coinbase_api/ml_models/rl_model.pkl'
        for crypto_symbol in self.crypto_environments:
            if os.path.exists(model_path):
                # Load the existing model
                print(f'Loading model for crypto: {crypto_symbol}')
                env = CustomEnv(data_handler=self.crypto_environments[crypto_symbol])
                model = PPO.load(model_path, env=env)
                self.crypto_models[crypto_symbol] = model
                print(f'Loaded model for crypto: {crypto_symbol}')
            else:
                raise NotImplementedError(f"PPO Model on path {model_path} does not exist. Real data application not available!")
            
    def get_action(self, crypto: AbstractOHLCV):
        model = self.crypto_models[crypto.symbol]
        vec_env = model.get_env()
        obs = vec_env.reset()
        action, states = model.predict(obs, deterministic=True)
        return action
    
    def get_all_actions(self):
        all_actions = {}
        for crypto_model in crypto_models:
            action = self.get_action(crypto_model)
            all_actions[crypto_model] = action[0] # collect actions
        return all_actions
