# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand, CommandError
# from celery_app.tasks import print_statement
from coinbase_api.constants import crypto_models, crypto_extra_features, crypto_features, crypto_predicted_features
import os
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from gymnasium import spaces
from coinbase_api.ml_models.RL_decider_model import CustomEnv
from coinbase_api.ml_models.data_handlers.real_data_handler import RealDataHandler
import numpy as np

class Command(BaseCommand):
    help = 'Test the real environment WITHOUT actually modifying the state'

    def handle(self, *args, **kwargs):
        data_handler = RealDataHandler()
        # action = [0.0 for i in range(len(crypto_models))]
        model_path = 'coinbase_api/ml_models/rl_model.pkl'
        # data_handler.update_state(action)
        if os.path.exists(model_path):
            # Load the existing model
            print('Loaded model!')
            env = CustomEnv(data_handler=data_handler)
            model = PPO.load(model_path, env=env)
        else:
            raise NotImplementedError(f"PPO Model on path {model_path} does not exist. Real data application not available!")
        vec_env = model.get_env()
        obs = vec_env.reset()
        print(f'observation before step: {obs}. {len(obs)}')
        for idx, entry in enumerate(obs[-1]):
            if (np.isnan(entry)):
                print(f'{idx}: {entry}')
        action, states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # obs, reward, done, info = model.env.step(obs)
        print(f'observation after step: {obs}. {len(obs)}')
        action, _ = model.predict(obs, deterministic=False)
        print(f'Action trying to take: {action}')