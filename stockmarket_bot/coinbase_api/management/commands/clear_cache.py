from django.core.management.base import BaseCommand
from django.core.cache import cache
# from your_app.models import YourModel  # Replace with your actual models

class Command(BaseCommand):
    help = 'Train the model and clear the cache after each interval'

    def handle(self, *args, **options):
        # Your training logic here
        # # for interval in training_intervals:
        #     self.train_interval(interval)
            # Clear the cache after the interval is completed
        cache.clear()
        self.stdout.write(self.style.SUCCESS('Cache cleared'))

    def train_interval(self, interval):
        # Training logic for each interval
        pass
