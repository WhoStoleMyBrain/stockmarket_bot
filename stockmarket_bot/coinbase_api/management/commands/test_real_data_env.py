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
from coinbase_api.ml_models.data_handlers.real_data_handler import RealDataHandler

class Command(BaseCommand):
    help = 'Test the real environment WITHOUT actually modifying the state'

    def handle(self, *args, **kwargs):
        data_handler = RealDataHandler()
        action = [0.0 for i in range(len(crypto_models))]
        data_handler.update_state(action)