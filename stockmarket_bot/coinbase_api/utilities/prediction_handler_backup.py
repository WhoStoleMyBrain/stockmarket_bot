from tqdm import tqdm
from ..constants import crypto_models
from coinbase_api.tasks import predict_with_lstm, predict_with_xgboost
from django.db.utils import IntegrityError
from django.db import transaction
from coinbase_api.enums import Database
from coinbase_api.models.models import AbstractOHLCV, Prediction
from concurrent.futures import ThreadPoolExecutor

class PredictionHandler:
    def __init__(self, lstm_sequence_length=100, database=Database.DEFAULT.value, timestamp=None) -> None:
        self.lstm_sequence_length = lstm_sequence_length
        self.timestamp = timestamp
        self.crypto_models = crypto_models
        self.data = None
        self.database = database
        
    def predict(self, *args, **kwargs):
        for crypto_model in self.crypto_models:
            self._predict_for_model(crypto_model)

    def _predict_for_model(self, crypto_model: AbstractOHLCV):
        if self.timestamp is None:
            self.data = crypto_model.objects.using(self.database).only('timestamp', 'volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns').order_by('-timestamp')[:self.lstm_sequence_length]
            self.timestamp = self.data.first().timestamp
        else:
            self.data = crypto_model.objects.using(self.database).filter(timestamp__lte=self.timestamp).only('timestamp', 'volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns').order_by('-timestamp')[:self.lstm_sequence_length]

        dataframe_lstm = crypto_model.queryset_to_lstm_dataframe(self.data)
        dataframe_xgboost = crypto_model.queryset_to_xgboost_dataframe(self.data[self.lstm_sequence_length-1:])
        # print(f'dataframe_lstm: {dataframe_lstm.shape}. timestamp: {self.timestamp}. crypto_model: {crypto_model}')
        self._try_predict(predict_with_lstm, 'lstm', {'data': dataframe_lstm, 'timestamp': self.timestamp, 'crypto_model': crypto_model, 'database': self.database})
        self._try_predict(predict_with_xgboost, 'XGBoost', {'data': dataframe_xgboost, 'timestamp': self.timestamp, 'crypto_model': crypto_model, 'database': self.database})

    def _try_predict(self, method, ml_model, kwargs):
        try:
            method(**kwargs)
        except IntegrityError:
            model = kwargs.get('crypto_model')
            timestamp = kwargs.get('timestamp')
            print(f'Prediction for {ml_model}, {model.__name__}, {timestamp} is already in DB.')

    def restore_prediction_database(self):
        print(f'Deleting all predictions in database: {self.database}')
        deleted_obj_count, _ = Prediction.objects.using(self.database).all().delete()
        print(f'Successfully deleted {deleted_obj_count} items from database: {self.database}')
    # def predict(self, *args, **kwargs):
    #     # Use ThreadPoolExecutor to parallelize predictions
    #     with ThreadPoolExecutor(max_workers=10) as executor:
    #         futures = [executor.submit(self._predict_for_model, crypto_model) for crypto_model in self.crypto_models]
    #         for future in futures:
    #             future.result()

    # def _predict_for_model(self, crypto_model: AbstractOHLCV):
    #     if self.timestamp is None:
    #         self.data = crypto_model.objects.using(self.database).all().order_by('-timestamp')[:self.lstm_sequence_length]
    #         self.timestamp = self.data.first().timestamp
    #     else:
    #         self.data = crypto_model.objects.using(self.database).filter(timestamp__lte=self.timestamp).order_by('-timestamp')[:self.lstm_sequence_length]

    #     dataframe_lstm = crypto_model.queryset_to_lstm_dataframe(self.data)
    #     dataframe_xgboost = crypto_model.queryset_to_xgboost_dataframe(self.data[self.lstm_sequence_length-1:])

    #     self._try_predict(predict_with_lstm, 'lstm', {'data': dataframe_lstm, 'timestamp': self.timestamp, 'crypto_model': crypto_model, 'database': self.database})
    #     self._try_predict(predict_with_xgboost, 'XGBoost', {'data': dataframe_xgboost, 'timestamp': self.timestamp, 'crypto_model': crypto_model, 'database': self.database})

    # def _try_predict(self, method, ml_model, kwargs):
    #     try:
    #         method(**kwargs)
    #     except IntegrityError:
    #         model = kwargs.get('crypto_model')
    #         timestamp = kwargs.get('timestamp')
    #         print(f'Prediction for {ml_model}, {model.__name__}, {timestamp} is already in DB.')

    def restore_prediction_database(self):
        print(f'Deleting all predictions in database: {self.database}')
        deleted_obj_count, _ = Prediction.objects.using(self.database).all().delete()
        print(f'Successfully deleted {deleted_obj_count} items from database: {self.database}')
