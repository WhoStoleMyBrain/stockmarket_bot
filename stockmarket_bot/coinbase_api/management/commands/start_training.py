# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand, CommandError
# from celery_app.tasks import print_statement
from coinbase_api.tasks import update_ohlcv_data
from coinbase_api.utilities.utils import cb_fetch_available_crypto
from coinbase_api.constants import crypto_models, crypto_extra_features, crypto_features, crypto_predicted_features
import os
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from gymnasium import spaces
from coinbase_api.ml_models.RL_decider_model import CustomEnv, SimulationDataHandler
import numpy as np

class Command(BaseCommand):
    help = 'Traing the RL model for a number of iterations over the whole dataset'

    def add_arguments(self, parser):
        parser.add_argument(
            '--iterations', 
            type=int, 
            help='Number of times to iterate over the whole dataset',
            required=False
        )

    def get_action_space(self):
        N = len(crypto_models)
        M = len(crypto_features) + len(crypto_predicted_features) + len(crypto_extra_features)
        shape_value = M*N + 2 #! +1 because of total volume held and USDC value held
        return spaces.Box(low=-np.inf, high=np.inf, shape=(shape_value*2,), dtype=np.float64)

    def handle(self, *args, **kwargs):
        param = kwargs.get('iterations', None)
        if param is not None:
            try:
                param = int(param)
            except ValueError:
                raise CommandError('The provided parameter is not a valid number')
        else:
            param = 50 # default value
        model_path = 'coinbase_api/ml_models/rl_model.pkl'
        # timesteps = 50
        data_handler = SimulationDataHandler()
        if os.path.exists(model_path):
            # Load the existing model
            print('Loaded model!')
            env = CustomEnv(data_handler=data_handler, total_steps=param, asymmetry_factor=0.5)
            model = PPO.load(model_path, env=env, n_steps=param, observation_space = self.get_action_space())
            model.action_space = self.get_action_space()
            model.observation_space = self.get_action_space()
            print(f'model obs shape: {model.observation_space.shape}')
        else:
            # Create a new model
            env = CustomEnv(data_handler=data_handler, total_steps=param, asymmetry_factor=0.5)
            model = PPO("MlpPolicy", env=env, verbose=0, n_steps=param)
            
        # env_checker.check_env(env)
        #? Model is loaded, now we need to set up for the training task.
        #? Most Importantly, the environment itself cannot step, since the data are externally driven
        # update_ohlcv_data()
        # cb_fetch_available_crypto()
        # cb_fetch_available_crypto_dummy()

        # obs = env.reset()  # Reset the environment to get the initial observation
        
        # action, _ = model.predict(obs, deterministic=True)
        # print(f'Action trying to take: {action}')
        print('starting training...')
        for _ in range(1):
        # for _ in range(10):
            try:
                model.learn(total_timesteps=param, progress_bar=True, reset_num_timesteps=True)
                model.save(model_path)
            except Exception as e:
                print(f'exception occured: {e}:{e.with_traceback()}')
                model.save(model_path)
        # obs, reward, done, info = env.step(action)
        # print(f'reward: {reward}')
        # obs, reward, done, truncated, info = env.step()
        # action, _ = model.predict(obs, deterministic=True)
        # model.save(model_path)

        # print(f'Action trying to take: {action}')
        # action, _ = model.predict(obs, deterministic=True)
        # model.learn(total_timesteps=1000)