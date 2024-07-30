# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand, CommandError
# from celery_app.tasks import print_statement
# from coinbase_api.utilities.utils import cb_fetch_available_crypto
from coinbase_api.constants import crypto_models, crypto_extra_features, crypto_features, crypto_predicted_features
import os
from stable_baselines3 import PPO
from gymnasium import spaces
from coinbase_api.ml_models.RL_decider_model import CustomEnv
import numpy as np
import traceback

from coinbase_api.ml_models.data_handlers.simulation_data_handler import SimulationDataHandler
from coinbase_api.ml_models.rl_model_logging_callback import RLModelLoggingCallback

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
            param = 100 # default value
        model_path = 'coinbase_api/ml_models/rl_model.pkl'
        log_dir = '/logs'
        # data_handler = SimulationDataHandler(total_steps=param)
        intervals = [168, 336, 504, 672]  # 1 week, 2 weeks, 3 weeks, 4 weeks
        # interval_weights = [2, 2, 2, 2]   # 4x1week, 3x2weeks, 2x3weeks, 1x4weeks
        interval_weights = [4, 3, 2, 1]   # 4x1week, 3x2weeks, 2x3weeks, 1x4weeks
        # intervals = [168]
        # interval_weights = [1]
        interval_transaction_costs = 0.0
        interval_list = [interval for interval, weight in zip(intervals, interval_weights) for _ in range(weight)]

        for interval in interval_list:
            print(f'Starting training with interval: {interval}')
            env = CustomEnv(data_handler=SimulationDataHandler(total_steps=interval, transaction_cost_factor=interval_transaction_costs))
            if os.path.exists(model_path):
                print('Loaded model!')
                # env = CustomEnv(data_handler=data_handler)
                model = PPO.load(model_path, env=env)
                model.tensorboard_log = log_dir
            else:
                # env = CustomEnv(data_handler=data_handler, total_steps=param, asymmetry_factor=0.5)
                model = PPO("MlpPolicy", env=env, verbose=0, tensorboard_log=log_dir)

                # model.set_parameters({'n_steps': interval})
            try:
                model.learn(total_timesteps=interval, progress_bar=True, reset_num_timesteps=True, tb_log_name=f"ModelV1_{interval}_{interval_transaction_costs}", log_interval=1, callback=RLModelLoggingCallback())
                model.save(model_path)
            except Exception as e:
                print(f'exception occurred: {e}: {traceback.format_exc()}')
                model.save(model_path)
        
        print(f'Training completed over intervals: {interval_list}')