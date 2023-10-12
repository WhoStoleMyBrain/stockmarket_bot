# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
from django_celery_beat.models import PeriodicTask, IntervalSchedule
# from celery_app.tasks import print_statement
from coinbase_api.tasks import update_ohlcv_data

class Command(BaseCommand):
    help = 'Setup periodic task'

    def handle(self, *args, **kwargs):
        update_ohlcv_data()