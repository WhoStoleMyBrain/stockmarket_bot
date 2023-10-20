# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand, CommandError
# from celery_app.tasks import print_statement
from coinbase_api.tasks import update_ohlcv_data
from coinbase_api.utilities.utils import cb_fetch_available_crypto
import os
from stable_baselines3 import PPO
from coinbase_api.ml_models.RL_decider_model import CustomEnv

class Command(BaseCommand):
    help = 'Traing the RL model for a number of iterations over the whole dataset'

    def add_arguments(self, parser):
        parser.add_argument(
            '--iterations', 
            type=int, 
            help='Number of times to iterate over the whole dataset',
            required=False
        )

    def handle(self, *args, **kwargs):
        param = kwargs.get('iterations', None)
        if param is not None:
            try:
                param = int(param)
            except ValueError:
                raise CommandError('The provided parameter is not a valid number')
        else:
            param = 2 # default value
        model_path = 'coinbase_api/ml_models/rl_model.pkl'
        
        if os.path.exists(model_path):
            # Load the existing model
            env = CustomEnv()
            model = PPO.load(model_path, env=env)
            # env = model.get_env()
        else:
            # Create a new model
            env = CustomEnv()
            model = PPO("MlpPolicy", env, verbose=1)
        # update_ohlcv_data()
        # cb_fetch_available_crypto()
        # cb_fetch_available_crypto_dummy()

        obs = env.reset()  # Reset the environment to get the initial observation
        
        action, _ = model.predict(obs, deterministic=True)
        print(f'Action trying to take: {action}')
        
        obs, reward, done, info = env.step(action)
        print(f'reward: {reward}')
        model.save(model_path)

        print(f'Action trying to take: {action}')
        action, _ = model.predict(obs, deterministic=True)
        # model.learn(total_timesteps=1000)