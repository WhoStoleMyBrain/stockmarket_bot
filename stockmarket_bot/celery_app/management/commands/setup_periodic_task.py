# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand
from django_celery_beat.models import PeriodicTask, IntervalSchedule
from base_app.tasks import print_statement

class Command(BaseCommand):
    help = 'Setup periodic task'

    def handle(self, *args, **kwargs):
        schedule, created = IntervalSchedule.objects.get_or_create(
            every=20,
            period=IntervalSchedule.SECONDS,
        )
        PeriodicTask.objects.get_or_create(
            interval=schedule,
            name='Print Hello every 20 seconds',
            task='base_app.tasks.print_statement',
        )
        self.stdout.write(self.style.SUCCESS('Successfully set up periodic task'))
