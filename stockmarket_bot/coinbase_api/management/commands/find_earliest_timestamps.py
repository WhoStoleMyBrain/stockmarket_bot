# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
from coinbase_api.utilities.utils import cb_find_earliest_data
from coinbase_api.constants import crypto_models
from datetime import datetime
from django.utils.timezone import make_aware, make_naive

from coinbase_api.models.models import CryptoMetadata



class Command(BaseCommand):
    help = 'Find earliest timestamps for all crypto models'

    def handle(self, *args, **kwargs):
        print('starting earliest timestamp')
        for model in crypto_models:
            try:
                obj = CryptoMetadata.objects.using('historical').get(
                    symbol=CryptoMetadata.symbol_to_storage(model.symbol),
                    )
                print(f'already had earliest data: {model.symbol}: {obj.earliest_date}')
                continue
            except CryptoMetadata.DoesNotExist:
                earliest_date_timestamp = cb_find_earliest_data(f'{model.symbol}-USDC')
            print(f'earliest_date_timestamp: {earliest_date_timestamp}')
            if earliest_date_timestamp is not None:
                earliest_date = datetime.fromtimestamp(earliest_date_timestamp)
                # Either create a new record or update the existing one
                print(CryptoMetadata.symbol_to_storage(model.symbol))
                try:
                    obj = CryptoMetadata.objects.using('historical').get(
                        symbol=CryptoMetadata.symbol_to_storage(model.symbol),
                        # earliest_date = earliest_date
                        )
                    if obj.earliest_date > earliest_date:
                        obj.earliest_date = earliest_date
                except CryptoMetadata.DoesNotExist:
                    obj = CryptoMetadata(
                        symbol=CryptoMetadata.symbol_to_storage(model.symbol),
                        earliest_date = earliest_date
                    )
                obj.save(using='historical')
                print(f'Stored the following date for {CryptoMetadata.symbol_to_storage(model.symbol)}: {obj.earliest_date}')

        