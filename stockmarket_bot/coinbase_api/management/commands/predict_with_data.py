# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
from coinbase_api.enums import Database
from coinbase_api.utilities.prediction_handler import PredictionHandler

class Command(BaseCommand):
    help = 'Predict with data'

    def handle(self, *args, **kwargs):
        print('starting predictions...')
        lstm_sequence_length = 100
        database = Database.DEFAULT.value
        prediction_handler = PredictionHandler(lstm_sequence_length=lstm_sequence_length, database=database)
        prediction_handler.predict()