from django.core.management.base import BaseCommand
from coinbase_api.enums import Database
from coinbase_api.models.generated_models import *
from coinbase_api.utilities.ml_utils import add_calculated_parameters
from coinbase_api.utilities.utils import fetch_hourly_data_for_crypto
from coinbase_api.constants import crypto_models

class Command(BaseCommand):
    help = 'Add calculated values to DB'

    def handle(self, *args, **kwargs):
        print('starting historical db update task')
        for crypto_model in crypto_models:
            print(f"starting to process crypto: {crypto_model.symbol}")
            add_calculated_parameters(crypto_model, database=Database.HISTORICAL.value)
        