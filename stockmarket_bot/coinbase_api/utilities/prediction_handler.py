from ..constants import crypto_models
from coinbase_api.tasks import predict_with_lstm, predict_with_xgboost
from django.db.utils import IntegrityError
from coinbase_api.enums import Database
from coinbase_api.models.models import AbstractOHLCV, Prediction
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import time
from django.db import transaction

class PredictionHandler:
    def __init__(self, lstm_sequence_length=100, database=Database.DEFAULT.value, timestamp=None) -> None:
        self.lstm_sequence_length = lstm_sequence_length
        self.timestamp = timestamp
        self.crypto_models = crypto_models
        self.database = database

    def predict(self, dataframes: dict[str, pd.DataFrame]):
        start_time = time.time()
        interval_times = {"Process LSTM Data": 0, "Process XGBoost Data": 0, "LSTM Prediction": 0, "XGBoost Prediction": 0}
        interval_counts = {"Process LSTM Data": 0, "Process XGBoost Data": 0, "LSTM Prediction": 0, "XGBoost Prediction": 0}
        predictions = []
        for crypto_model in self.crypto_models:
            symbol_data = dataframes[crypto_model.symbol]
            if len(symbol_data) < self.lstm_sequence_length:
                print(f"Not enough data for {crypto_model.symbol}")
                continue

            interval_start = time.time()
            dataframe_lstm = self.process_lstm_data(symbol_data)
            interval_times["Process LSTM Data"] += time.time() - interval_start
            interval_counts["Process LSTM Data"] += 1

            interval_start = time.time()
            dataframe_xgboost = self.process_xgboost_data(symbol_data.iloc[self.lstm_sequence_length-1:])
            interval_times["Process XGBoost Data"] += time.time() - interval_start
            interval_counts["Process XGBoost Data"] += 1

            interval_start = time.time()
            
            self._try_predict(predict_with_lstm, 'lstm', {'data': dataframe_lstm, 'timestamp': self.timestamp, 'crypto_model': crypto_model, 'predictions': predictions, 'database': self.database})
            interval_times["LSTM Prediction"] += time.time() - interval_start
            interval_counts["LSTM Prediction"] += 1

            interval_start = time.time()
            self._try_predict(predict_with_xgboost, 'XGBoost', {'data': dataframe_xgboost, 'timestamp': self.timestamp, 'crypto_model': crypto_model, 'predictions': predictions, 'database': self.database})
            interval_times["XGBoost Prediction"] += time.time() - interval_start
            interval_counts["XGBoost Prediction"] += 1
        interval_start = time.time()
        interval_counts["Updating db"] = 1
        with transaction.atomic(using=self.database):
            Prediction.objects.using(self.database).bulk_create(predictions)
        interval_times["Updating db"] = time.time() - interval_start
            

        end_time = time.time()
        total_time = end_time - start_time

        # Print timing information
        print(f'Total time for predict: {total_time:.2f} seconds')
        for interval_name, interval_duration in interval_times.items():
            avg_time = interval_duration / interval_counts[interval_name] if interval_counts[interval_name] > 0 else 0
            print(f'{interval_name}: {interval_duration:.2f} seconds ({(interval_duration / total_time) * 100:.2f}%) - Average time: {avg_time:.4f} seconds')

    def _try_predict(self, method, ml_model, kwargs):
        try:
            method(**kwargs)
        except IntegrityError:
            model = kwargs.get('crypto_model')
            timestamp = kwargs.get('timestamp')
            print(f'Prediction for {ml_model}, {model.__name__}, {timestamp} is already in DB.')

    def process_lstm_data(self, dataframe: pd.DataFrame):
        features = ['volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns']
        dataframe = dataframe[features].fillna(0)
        
        prices = dataframe.values
        scaler = StandardScaler()
        prices_scaled = scaler.fit_transform(prices)
        
        tensor_data = torch.tensor(prices_scaled, dtype=torch.float32).unsqueeze(0)
        return tensor_data

    def process_xgboost_data(self, dataframe: pd.DataFrame):
        features = ['volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns']
        dataframe = dataframe[features].fillna(0)
        
        dataframe['Hour'] = dataframe.index.hour
        dataframe['Day_of_Week'] = dataframe.index.dayofweek
        dataframe['Day_of_Month'] = dataframe.index.day
        dataframe['Month'] = dataframe.index.month
        dataframe['Year'] = dataframe.index.year
        dataframe['Is_Weekend'] = (dataframe['Day_of_Week'] >= 5).astype(int)
        
        features_extended = AbstractOHLCV.get_features_dropped()
        dataframe_extended = dataframe[features_extended]
        
        scaler = StandardScaler()
        dataframe_scaled = scaler.fit_transform(dataframe_extended)
        data_dmatrix = xgb.DMatrix(dataframe_scaled)
        return data_dmatrix

    def restore_prediction_database(self):
        print(f'Deleting all predictions in database: {self.database}')
        deleted_obj_count, _ = Prediction.objects.using(self.database).all().delete()
        print(f'Successfully deleted {deleted_obj_count} items from database: {self.database}')
