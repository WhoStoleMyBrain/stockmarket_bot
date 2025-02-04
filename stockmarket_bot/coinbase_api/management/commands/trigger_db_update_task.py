# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
# from celery_app.tasks import print_statement
from coinbase_api.tasks import update_ohlcv_data
import os
from stable_baselines3 import PPO
from coinbase_api.ml_models.RL_decider_model import CustomEnv

class Command(BaseCommand):
    help = 'Trigger Database Update Task'

    def handle(self, *args, **kwargs):
        update_ohlcv_data()
