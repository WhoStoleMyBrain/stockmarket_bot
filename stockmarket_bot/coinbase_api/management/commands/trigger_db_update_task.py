# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
# from celery_app.tasks import print_statement
from coinbase_api.tasks import update_ohlcv_data
from coinbase_api.utilities.utils import cb_fetch_available_crypto
import os
from stable_baselines3 import PPO
from coinbase_api.ml_models.RL_decider_model import CustomEnv

class Command(BaseCommand):
    help = 'Trigger Database Update Task'

    def handle(self, *args, **kwargs):
        # model_path = 'coinbase_api/ml_models/rl_model.pkl'
        
        # if os.path.exists(model_path):
        #     # Load the existing model
        #     env = CustomEnv()
        #     model = PPO.load(model_path, env=env)
        #     # env = model.get_env()
        # else:
        #     # Create a new model
        #     env = CustomEnv()
        #     model = PPO("MlpPolicy", env, verbose=1)
        update_ohlcv_data()
        # cb_fetch_available_crypto()
        # cb_fetch_available_crypto_dummy()

        # obs = env.reset()  # Reset the environment to get the initial observation
        
        # action, _ = model.predict(obs, deterministic=True)
        # print(f'Action trying to take: {action}')
        
        # obs, reward, done, info = env.step(action)
        # print(f'reward: {reward}')
        # model.save(model_path)

        # print(f'Action trying to take: {action}')
        # action, _ = model.predict(obs, deterministic=True)
        # model.learn(total_timesteps=1000)