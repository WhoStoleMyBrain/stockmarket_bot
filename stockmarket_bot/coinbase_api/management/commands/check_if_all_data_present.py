# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
# from django_celery_beat.models import PeriodicTask, IntervalSchedule
# from celery_app.tasks import print_statement
from coinbase_api.models.models import Prediction
# from coinbase_api.tasks import predict_with_lstm, predict_with_xgboost
from datetime import datetime, timezone, timedelta
from django.utils.timezone import make_aware

class Command(BaseCommand):
    help = 'Check if all data are present'

    def handle(self, *args, **kwargs):
        timestamp = make_aware(datetime.now().replace(hour=0, minute=0, second=0))
        predictions = Prediction.objects.filter(timestamp_predicted_for__gte=timestamp)
        count = predictions.count()
        if count == 0:
            print('did not find any predictions')
        else:
            print('found the following predictions:')
            for item in predictions:
                print(f'item: {item.crypto}, {item.model_name}')
        # data_btc = Bitcoin.objects.all()
        # data_ethereum = Ethereum.objects.all()
        # data_polkadot = Polkadot.objects.all()
        