from ..constants import crypto_models
from coinbase_api.tasks import predict_with_lstm, predict_with_xgboost
from django.db.utils import IntegrityError
from coinbase_api.enums import Database
from coinbase_api.models.models import Prediction

class PredictionHandler:
    def __init__(self, lstm_sequence_length=100, database = Database.DEFAULT.value, timestamp=None) -> None:
        self.lstm_sequence_length = lstm_sequence_length
        self.timestamp = timestamp
        self.crypto_models = crypto_models
        self.data = None
        self.database = database

    def predict(self, *args, **kwargs):
        # print('starting predictions...')
        for crypto_model in self.crypto_models:
            # print(f'starting with model: {crypto_model.symbol}')
            # print(f'self.timestamp: {self.timestamp}')

            if self.timestamp is None:
                self.data = crypto_model.objects.using(self.database).all().order_by('-timestamp')[:self.lstm_sequence_length]
                self.timestamp = self.data.first().timestamp
            else:
                self.data = crypto_model.objects.using(self.database).filter(timestamp__lte=self.timestamp).order_by('-timestamp')[:self.lstm_sequence_length]
                # tmp1 = crypto_model.objects.using(self.database).filter(timestamp__lte=self.timestamp).order_by('timestamp').first()
                # tmp2 = crypto_model.objects.using(self.database).filter(timestamp__lte=self.timestamp).order_by('-timestamp').first()
                # print(f'first entry: {tmp2.timestamp}, last entry: {tmp1.timestamp}')
            dataframe_lstm = crypto_model.queryset_to_lstm_dataframe(self.data)
            dataframe_xgboost = crypto_model.queryset_to_xgboost_dataframe(self.data[self.lstm_sequence_length-1:])
            self._try_predict(predict_with_lstm, 'lstm', {'data':dataframe_lstm,'timestamp':self.timestamp, 'crypto_model':crypto_model, 'database':self.database})
            self._try_predict(predict_with_xgboost, 'XGBoost', {'data':dataframe_xgboost,'timestamp':self.timestamp, 'crypto_model':crypto_model, 'database':self.database})
        # print('finished predictions...')

    def _try_predict(self, method, ml_model, kwargs):
        try:
            method(**kwargs)
        except IntegrityError:
            try:
                model = kwargs['crypto_model']
                timestamp = kwargs['timestamp']
            except KeyError:
                print(f'Could not find relevant info inside kwargs: {kwargs}')
                return
            print(f'Prediction for {ml_model}, {model.__name__}, {timestamp} is already in DB.')

    def restore_prediction_database(self):
        print(f'Deleting all predictions in database: {self.database}')
        all_predictions = Prediction.objects.using(self.database).all()
        # prediction_count = len(all_predictions)
        deleted_obj_count, _ = all_predictions.delete()
        print(f'Successfully deleted {deleted_obj_count} items from database: {self.database}')
        