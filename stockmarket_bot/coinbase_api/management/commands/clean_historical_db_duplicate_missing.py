# base_app/management/commands/clean_historical_data.py
import logging
from django.core.management.base import BaseCommand
from django.db.models import Count
from coinbase_api.constants import crypto_models
from coinbase_api.enums import Database
from datetime import datetime, timedelta

from coinbase_api.models.generated_models import *
from coinbase_api.models.models import AbstractOHLCV

class Command(BaseCommand):
    help = 'Clean historical data by removing duplicates and handling missing data.'

    def __init__(self):
        super().__init__()
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler for logging
        handler = logging.FileHandler('clean_historical_data.log')
        handler.setLevel(logging.DEBUG)

        # Create a console handler for logging
        console_handler = logging.StreamHandler(self.stdout)
        console_handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(handler)
        self.logger.addHandler(console_handler)

    def handle(self, *args, **kwargs):
        main_crypto_models:list[AbstractOHLCV] = [BTC, ETH]
        for crypto_model in main_crypto_models:
            message = f"Processing {crypto_model.symbol}"
            self.logger.info(message)
            
            # Step 1: Remove duplicate entries
            self.remove_duplicates(crypto_model)
            
            # Step 2: Handle missing data
            self.handle_missing_data(crypto_model)
            
            message = f"Finished processing {crypto_model.symbol}"
            self.logger.info(message)

    def remove_duplicates(self, crypto_model:AbstractOHLCV):
        duplicates = (crypto_model.objects.using(Database.HISTORICAL.value)
                      .values('timestamp')
                      .annotate(count=Count('id'))
                      .filter(count__gt=1))

        for duplicate in duplicates:
            timestamp = duplicate['timestamp']
            records = list(crypto_model.objects.using(Database.HISTORICAL.value).filter(timestamp=timestamp))
            # Keep the first record and delete the others
            for record in records[1:]:
                record.delete()
            message = f"Removed duplicates for {crypto_model.symbol} at {timestamp}"
            self.logger.info(message)

    def handle_missing_data(self, crypto_model:AbstractOHLCV):
        all_records = (crypto_model.objects.using(Database.HISTORICAL.value)
                       .order_by('timestamp'))

        previous_record = None
        for record in all_records:
            if previous_record and (record.timestamp - previous_record.timestamp) > timedelta(minutes=5):
                # Fill missing timestamps
                missing_timestamp = previous_record.timestamp + timedelta(minutes=5)
                while missing_timestamp < record.timestamp:
                    # Duplicate the previous record with the missing timestamp
                    crypto_model.objects.using(Database.HISTORICAL.value).create(
                        timestamp=missing_timestamp,
                        open=previous_record.open,
                        high=previous_record.high,
                        low=previous_record.low,
                        close=previous_record.close,
                        volume=previous_record.volume,
                        sma=previous_record.sma,
                        ema=previous_record.ema,
                        rsi=previous_record.rsi,
                        macd=previous_record.macd,
                        bollinger_high=previous_record.bollinger_high,
                        bollinger_low=previous_record.bollinger_low,
                        vmap=previous_record.vmap,
                        percentage_returns=previous_record.percentage_returns,
                        log_returns=previous_record.log_returns,
                    )
                    message = f"Filled missing data for {crypto_model.symbol} at {missing_timestamp}"
                    self.logger.info(message)
                    missing_timestamp += timedelta(minutes=5)

            previous_record = record

