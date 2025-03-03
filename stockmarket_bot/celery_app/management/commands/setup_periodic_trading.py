# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
from django_celery_beat.models import PeriodicTask, IntervalSchedule

class Command(BaseCommand):
    help = 'Setup periodic trading'

    def handle(self, *args, **kwargs):
        schedule, created = IntervalSchedule.objects.get_or_create(
            every=5,
            period=IntervalSchedule.MINUTES,
        )
        PeriodicTask.objects.get_or_create(
            interval=schedule,
            name='Print Hello every 20 seconds',
            task='celery_app.tasks.run_trading_bot',
        )
        self.stdout.write(self.style.SUCCESS('Successfully set up periodic trading task'))
