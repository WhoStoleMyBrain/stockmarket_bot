# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
# from django_celery_beat.models import PeriodicTask, IntervalSchedule
# from celery_app.tasks import print_statement
# from coinbase_api.tasks import update_ohlcv_data
from coinbase_api.models import Bitcoin, Ethereum, Polkadot
from coinbase_api.tasks import predict_with_lstm, predict_with_xgboost

class Command(BaseCommand):
    help = 'Predict with data'

    def handle(self, *args, **kwargs):
        print('starting predictions...')
        lstm_sequence_length = 100
        data_btc = Bitcoin.objects.all().order_by('-timestamp')[:lstm_sequence_length]
        data_ethereum = Ethereum.objects.all().order_by('-timestamp')[:lstm_sequence_length]
        data_polkadot = Polkadot.objects.all().order_by('-timestamp')[:lstm_sequence_length]
        dataframe_btc = Bitcoin.queryset_to_lstm_dataframe(data_btc)
        dataframe_ethereum = Bitcoin.queryset_to_lstm_dataframe(data_ethereum)
        dataframe_polkadot = Bitcoin.queryset_to_lstm_dataframe(data_polkadot)
        dataframe_xgboost_btc = Bitcoin.queryset_to_xgboost_dataframe(data_btc[lstm_sequence_length-1:])
        dataframe_xgboost_ethereum = Bitcoin.queryset_to_xgboost_dataframe(data_ethereum[lstm_sequence_length-1:])
        dataframe_xgboost_polkadot = Bitcoin.queryset_to_xgboost_dataframe(data_polkadot[lstm_sequence_length-1:])
        
        predict_with_lstm(dataframe_btc)
        predict_with_lstm(dataframe_ethereum)
        predict_with_lstm(dataframe_polkadot)
        predict_with_xgboost(dataframe_xgboost_btc)
        predict_with_xgboost(dataframe_xgboost_ethereum)
        predict_with_xgboost(dataframe_xgboost_polkadot)
        print('finished predictions...')