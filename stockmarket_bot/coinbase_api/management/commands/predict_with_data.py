# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
from django.db.utils import IntegrityError
# from django_celery_beat.models import PeriodicTask, IntervalSchedule
# from celery_app.tasks import print_statement
# from coinbase_api.tasks import update_ohlcv_data
from coinbase_api.models.models import Bitcoin, Ethereum, Polkadot, Prediction
from coinbase_api.tasks import predict_with_lstm, predict_with_xgboost

class Command(BaseCommand):
    help = 'Predict with data'

    def handle(self, *args, **kwargs):
        print('starting predictions...')
        # all_predictions = Prediction.objects.all()
        # all_predictions.delete()
        lstm_sequence_length = 100
        data_btc = Bitcoin.objects.all().order_by('-timestamp')[:lstm_sequence_length]
        data_ethereum = Ethereum.objects.all().order_by('-timestamp')[:lstm_sequence_length]
        data_polkadot = Polkadot.objects.all().order_by('-timestamp')[:lstm_sequence_length]
        timestamp_btc = data_btc.first().timestamp
        timestamp_ethereum = data_ethereum.first().timestamp
        timestamp_polkadot = data_polkadot.first().timestamp
        # print(timestamp_btc)
        dataframe_btc = Bitcoin.queryset_to_lstm_dataframe(data_btc)
        dataframe_ethereum = Bitcoin.queryset_to_lstm_dataframe(data_ethereum)
        dataframe_polkadot = Bitcoin.queryset_to_lstm_dataframe(data_polkadot)
        dataframe_xgboost_btc = Bitcoin.queryset_to_xgboost_dataframe(data_btc[lstm_sequence_length-1:])
        dataframe_xgboost_ethereum = Bitcoin.queryset_to_xgboost_dataframe(data_ethereum[lstm_sequence_length-1:])
        dataframe_xgboost_polkadot = Bitcoin.queryset_to_xgboost_dataframe(data_polkadot[lstm_sequence_length-1:])
        self.try_predict(predict_with_lstm, 'lstm', {'data':dataframe_btc,'timestamp':timestamp_btc, 'crypto_model':Bitcoin})
        self.try_predict(predict_with_lstm, 'lstm', {'data':dataframe_ethereum,'timestamp':timestamp_ethereum, 'crypto_model':Ethereum})
        self.try_predict(predict_with_lstm, 'lstm', {'data':dataframe_polkadot,'timestamp':timestamp_polkadot, 'crypto_model':Polkadot})
        self.try_predict(predict_with_xgboost, 'XGBoost', {'data':dataframe_xgboost_btc,'timestamp':timestamp_btc, 'crypto_model':Bitcoin})
        self.try_predict(predict_with_xgboost, 'XGBoost', {'data':dataframe_xgboost_ethereum,'timestamp':timestamp_ethereum, 'crypto_model':Ethereum})
        self.try_predict(predict_with_xgboost, 'XGBoost', {'data':dataframe_xgboost_polkadot,'timestamp':timestamp_polkadot, 'crypto_model':Polkadot})
        print('finished predictions...')

    def try_predict(self, method, ml_model, kwargs):
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