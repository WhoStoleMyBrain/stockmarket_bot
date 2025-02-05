# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
from coinbase_api.constants import crypto_models
from coinbase_api.enums import Database
from django.db.models import Q

class Command(BaseCommand):
    help = 'Replace all known null values in historical db'

    def handle(self, *args, **kwargs):

        print('starting replacement on db')
        for crypto_model in crypto_models:
            
            non_finished_objects = crypto_model.objects.using(Database.HISTORICAL.value).filter(
                Q(sma__isnull=True) |
                Q(ema__isnull=True) |
                Q(percentage_returns__isnull=True) |
                Q(log_returns__isnull=True) |
                Q(rsi__isnull=True) |
                Q(bollinger_low__isnull=True) |
                Q(bollinger_high__isnull=True) |
                Q(macd__isnull=True)
            )
            print(f'{crypto_model.symbol} count: {non_finished_objects.count()}')
            if (len(non_finished_objects) > 250):
                for i in range(int(len(non_finished_objects)/250)):
                    # print(f'idx: {i}')
                    tmp_items = non_finished_objects[i*250:(i+1)*250]
                    finished_objects = [item.set_nonset_values_to_default() for item in tmp_items]
                    crypto_model.objects.using(Database.HISTORICAL.value).bulk_update(finished_objects, fields=[
                        'vmap', 'percentage_returns', 'log_returns', 'open', 'high', 'low', 'close', 'volume',
                        'close_higher_shifted_1h', 'close_higher_shifted_24h', 'close_higher_shifted_168h',
                        'sma', 'ema', 'macd', 'bollinger_high', 'bollinger_low', 'rsi'
                    ])
            else:
                finished_objects = [item.set_nonset_values_to_default() for item in non_finished_objects]
                crypto_model.objects.using(Database.HISTORICAL.value).bulk_update(finished_objects, fields=[
                    'vmap', 'percentage_returns', 'log_returns', 'open', 'high', 'low', 'close', 'volume',
                    'close_higher_shifted_1h', 'close_higher_shifted_24h', 'close_higher_shifted_168h',
                    'sma', 'ema', 'macd', 'bollinger_high', 'bollinger_low', 'rsi'
                ])
            