from ..constants import crypto_models
from coinbase_api.tasks import predict_with_lstm, predict_with_xgboost
from django.db.utils import IntegrityError
from coinbase_api.enums import Database
from coinbase_api.models.models import AbstractOHLCV, Prediction
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class PredictionHandler:
    def __init__(self, lstm_sequence_length=100, database=Database.DEFAULT.value, timestamp=None) -> None:
        self.lstm_sequence_length = lstm_sequence_length
        self.timestamp = timestamp
        self.crypto_models = crypto_models
        self.database = database
        self.dataframes = self.initialize_dataframes()

    def initialize_dataframes(self) -> dict[str, pd.DataFrame]:
        dataframes = {}
        start_timestamp = self.timestamp - timedelta(hours=self.lstm_sequence_length - 1)
        complete_index = pd.date_range(start=start_timestamp, end=self.timestamp, freq='H')
        columns = ['volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns']

        for crypto_model in self.crypto_models:
            data = crypto_model.objects.using(self.database).filter(
                timestamp__gte=start_timestamp,
                timestamp__lte=self.timestamp
            ).values(
                'timestamp', 'volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns'
            ).order_by('timestamp')

            df = pd.DataFrame(data)
            if 'timestamp' not in df.columns or df.empty:
                # Create an empty dataframe with the required columns if there's no data
                df = pd.DataFrame(columns=['timestamp'] + columns)
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df[~df.index.duplicated(keep='first')]  # Remove duplicate timestamps

            # Reindex to include all timestamps and fill missing values
            df = df.reindex(complete_index)  # Reindex to include all timestamps
            df['symbol'] = crypto_model.symbol
            df['symbol'].fillna(crypto_model.symbol, inplace=True)
            df.fillna(0, inplace=True)  # Fill missing values with 0
            dataframes[crypto_model.symbol] = df

        return dataframes


    def update_dataframes(self):
        for crypto_model in self.crypto_models:
            new_data = crypto_model.objects.using(self.database).filter(
                timestamp=self.timestamp
            ).values(
                'timestamp', 'volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns'
            ).first()
            if new_data:
                new_data['symbol'] = crypto_model.symbol
                new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
                new_df = pd.DataFrame([new_data])
                new_df.set_index('timestamp', inplace=True)
                new_df.sort_index(inplace=True)
                existing_df = self.dataframes[crypto_model.symbol]
                combined_df = pd.concat([existing_df, new_df]).sort_index().iloc[-self.lstm_sequence_length:]
                self.dataframes[crypto_model.symbol] = combined_df

    def predict(self, *args, **kwargs):
        self.update_dataframes()
        
        for crypto_model in self.crypto_models:
            symbol_data = self.dataframes[crypto_model.symbol]
            if len(symbol_data) < self.lstm_sequence_length:
                print(f"Not enough data for {crypto_model.symbol}")
                continue
            dataframe_lstm = self.process_lstm_data(symbol_data)
            dataframe_xgboost = self.process_xgboost_data(symbol_data.iloc[self.lstm_sequence_length-1:])
            self._try_predict(predict_with_lstm, 'lstm', {'data': dataframe_lstm, 'timestamp': self.timestamp, 'crypto_model': crypto_model, 'database': self.database})
            self._try_predict(predict_with_xgboost, 'XGBoost', {'data': dataframe_xgboost, 'timestamp': self.timestamp, 'crypto_model': crypto_model, 'database': self.database})
            
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
