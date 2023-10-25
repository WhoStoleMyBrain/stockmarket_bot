# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
# from celery_app.tasks import print_statement
from coinbase_api.utilities.utils import fetch_hourly_data_for_crypto
import os
from stable_baselines3 import PPO
from coinbase_api.ml_models.RL_decider_model import CustomEnv
from constants import crypto_models

class Command(BaseCommand):
    help = 'Trigger Historical Database Update Task'

    def handle(self, *args, **kwargs):
        for crypto_model in crypto_models:
            fetch_hourly_data_for_crypto(crypto_model)
        