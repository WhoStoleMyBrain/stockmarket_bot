# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
from django.db.utils import IntegrityError
# from django_celery_beat.models import PeriodicTask, IntervalSchedule
# from celery_app.tasks import print_statement
# from coinbase_api.tasks import update_ohlcv_data
# from coinbase_api.models.models import Bitcoin, Ethereum, Polkadot, Prediction
from coinbase_api.tasks import predict_with_lstm, predict_with_xgboost
from constants import crypto_models
from enums import Database
from coinbase_api.utilities.prediction_handler import PredictionHandler

class Command(BaseCommand):
    help = 'Predict with data'

    def handle(self, *args, **kwargs):
        print('starting predictions...')
        # all_predictions = Prediction.objects.all()
        # all_predictions.delete()
        lstm_sequence_length = 100
        database = Database.DEFAULT.value
        prediction_handler = PredictionHandler(lstm_sequence_length=lstm_sequence_length, database=database)
        prediction_handler.predict()
    #     for crypto_model in crypto_models:
    #         data = crypto_model.objects.all().order_by('-timestamp')[:lstm_sequence_length]
    #         timestamp = data.first().timestamp
    #         dataframe_lstm = crypto_model.queryset_to_lstm_dataframe(data)
    #         dataframe_xgboost = crypto_model.queryset_to_xgboost_dataframe(data[lstm_sequence_length-1:])
    #         self.try_predict(predict_with_lstm, 'lstm', {'data':dataframe_lstm,'timestamp':timestamp, 'crypto_model':crypto_model})
    #         self.try_predict(predict_with_xgboost, 'XGBoost', {'data':dataframe_xgboost,'timestamp':timestamp, 'crypto_model':crypto_model})
    #     print('finished predictions...')

    # def try_predict(self, method, ml_model, kwargs):
    #     try:
    #         method(**kwargs)
    #     except IntegrityError:
    #         try:
    #             model = kwargs['crypto_model']
    #             timestamp = kwargs['timestamp']
    #         except KeyError:
    #             print(f'Could not find relevant info inside kwargs: {kwargs}')
    #             return
    #         print(f'Prediction for {ml_model}, {model.__name__}, {timestamp} is already in DB.')