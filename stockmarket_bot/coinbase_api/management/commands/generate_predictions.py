import time
import tqdm
import pandas as pd
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from coinbase_api.models.models import Prediction, AbstractOHLCV
from coinbase_api.enums import Database
from coinbase_api.constants import crypto_models
from coinbase_api.tasks import predict_with_lstm, predict_with_xgboost
from coinbase_api.utilities.prediction_handler import PredictionHandler
from django.db.utils import IntegrityError
from django.db import transaction

class Command(BaseCommand):
    help = 'Generate predictions for all possible timestamps for each cryptocurrency'

    def add_arguments(self, parser):
        parser.add_argument('--database', type=str, default=Database.HISTORICAL.value, help='Database to use for fetching historical data')

    def handle(self, *args, **options):
        database = options['database']
        lstm_sequence_length = 100

        for crypto_model in crypto_models:
            print(f"Processing {crypto_model.symbol}...")

            # Fetch historical data
            historical_data = crypto_model.objects.using(database).all().order_by('timestamp')
            if len(historical_data) < lstm_sequence_length:
                print(f"Not enough data for {crypto_model.symbol}")
                continue

            # Convert to DataFrame
            df = pd.DataFrame(list(historical_data.values()))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Initialize prediction handler
            prediction_handler = PredictionHandler(lstm_sequence_length=lstm_sequence_length, database=database, timestamp=df.index[-1])
            
            # Get all timestamps for predictions
            start_timestamp = df.index[lstm_sequence_length - 1]
            end_timestamp = df.index[-1]
            total_predictions = len(df) - lstm_sequence_length + 1

            print(f"Total predictions to make for {crypto_model.symbol}: {total_predictions}")
            print(f"Start timestamp of predictions: {start_timestamp}")
            print(f"End timestamp of predictions: {end_timestamp}")

            # Prepare predictions
            predictions = []
            total_lstm_time = 0
            total_xgboost_time = 0
            for i in tqdm.tqdm(range(total_predictions)):
                timestamp = df.index[i + lstm_sequence_length - 1]
                symbol_data = df.iloc[i:i + lstm_sequence_length]

                # Process LSTM data
                lstm_start_time = time.time()
                dataframe_lstm = prediction_handler.process_lstm_data(symbol_data)
                lstm_time = time.time() - lstm_start_time
                total_lstm_time += lstm_time

                # Process XGBoost data
                xgboost_start_time = time.time()
                dataframe_xgboost = prediction_handler.process_xgboost_data(symbol_data.iloc[-1:])
                xgboost_time = time.time() - xgboost_start_time
                total_xgboost_time += xgboost_time

                # Make predictions
                prediction_handler._try_predict(predict_with_lstm, 'lstm', {'data': dataframe_lstm, 'timestamp': timestamp, 'crypto_model': crypto_model, 'predictions': predictions, 'database': database})
                prediction_handler._try_predict(predict_with_xgboost, 'XGBoost', {'data': dataframe_xgboost, 'timestamp': timestamp, 'crypto_model': crypto_model, 'predictions': predictions, 'database': database})

            # Bulk create predictions
            with transaction.atomic(using=database):
                Prediction.objects.using(database).bulk_create(predictions)

            # Output timing information
            avg_lstm_time = total_lstm_time / total_predictions
            avg_xgboost_time = total_xgboost_time / total_predictions
            print(f"Total time for LSTM predictions for {crypto_model.symbol}: {total_lstm_time:.4f} seconds")
            print(f"Average time per LSTM prediction for {crypto_model.symbol}: {avg_lstm_time:.4f} seconds")
            print(f"Total time for XGBoost predictions for {crypto_model.symbol}: {total_xgboost_time:.4f} seconds")
            print(f"Average time per XGBoost prediction for {crypto_model.symbol}: {avg_xgboost_time:.4f} seconds")

            print(f"Finished processing {crypto_model.symbol}")

        print("All predictions generated successfully.")
