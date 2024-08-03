import os
import time
import tqdm
import pandas as pd
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from coinbase_api.models.models import AbstractOHLCV
from coinbase_api.enums import Database
from coinbase_api.constants import crypto_models
from tensorboardX import SummaryWriter

class Command(BaseCommand):
    help = 'Log best, worst, and average performance metrics to TensorBoard'

    def add_arguments(self, parser):
        parser.add_argument('--logdir', type=str, default='runs', help='Directory to save TensorBoard logs')
        parser.add_argument('--database', type=str, default=Database.HISTORICAL.value, help='Database to use for fetching historical data')

    def handle(self, *args, **options):
        logdir = options['logdir']
        database = options['database']
        
        writer_best = SummaryWriter(os.path.join(logdir, 'best'))
        writer_worst = SummaryWriter(os.path.join(logdir, 'worst'))
        writer_average = SummaryWriter(os.path.join(logdir, 'average'))

        earliest_timestamp = self.get_earliest_timestamp()
        latest_timestamp = self.get_latest_timestamp()

        total_steps = int((latest_timestamp - earliest_timestamp).total_seconds() / 3600) + 1
        initial_volume = 1000.0

        for step in tqdm.tqdm(range(total_steps)):
            current_timestamp = earliest_timestamp + timedelta(hours=step)
            best_performance, worst_performance, avg_performance = self.calculate_performance(current_timestamp, initial_volume, database)
            
            writer_best.add_scalar('Total Volume', best_performance, step)
            writer_worst.add_scalar('Total Volume', worst_performance, step)
            writer_average.add_scalar('Total Volume', avg_performance, step)

        writer_best.close()
        writer_worst.close()
        writer_average.close()

    def get_earliest_timestamp(self) -> datetime:
        timestamps = AbstractOHLCV.objects.using(Database.HISTORICAL.value).values_list('timestamp', flat=True).order_by('timestamp')
        return timestamps.first()

    def get_latest_timestamp(self) -> datetime:
        timestamps = AbstractOHLCV.objects.using(Database.HISTORICAL.value).values_list('timestamp', flat=True).order_by('-timestamp')
        return timestamps.first()

    def calculate_performance(self, timestamp: datetime, initial_volume: float, database: str) -> Tuple[float, float, float]:
        volumes = []
        for crypto_model in crypto_models:
            try:
                crypto_data = crypto_model.objects.using(database).get(timestamp=timestamp)
                volumes.append(crypto_data.close)
            except crypto_model.DoesNotExist:
                continue
        
        if not volumes:
            return initial_volume, initial_volume, initial_volume

        best_performance = initial_volume * max(volumes)
        worst_performance = initial_volume * min(volumes)
        avg_performance = initial_volume * (sum(volumes) / len(volumes))

        return best_performance, worst_performance, avg_performance
