# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
from coinbase_api.utilities.utils import cb_find_earliest_data
from constants import crypto_models
from datetime import datetime

from coinbase_api.models.models import CryptoMetadata



class Command(BaseCommand):
    help = 'Find earliest timestamps for all crypto models'

    def handle(self, *args, **kwargs):
        for model in crypto_models:
            earliest_date_timestamp = cb_find_earliest_data(f'{model.symbol}-USDC')
            if earliest_date_timestamp:
                earliest_date = datetime.fromtimestamp(earliest_date_timestamp)
                # Either create a new record or update the existing one
                CryptoMetadata.objects.update_or_create(
                    symbol=model.symbol,
                    defaults={'earliest_date': earliest_date}
                )

        