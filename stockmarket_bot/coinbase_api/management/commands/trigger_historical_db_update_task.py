from django.core.management.base import BaseCommand
from coinbase_api.utilities.utils import fetch_hourly_data_for_crypto
from coinbase_api.constants import crypto_models

class Command(BaseCommand):
    help = 'Trigger Historical Database Update Task'

    def handle(self, *args, **kwargs):
        print('starting historical db update task')
        for crypto_model in crypto_models:
            fetch_hourly_data_for_crypto(crypto_model)
        