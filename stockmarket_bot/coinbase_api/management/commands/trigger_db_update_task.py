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
        update_ohlcv_data()
