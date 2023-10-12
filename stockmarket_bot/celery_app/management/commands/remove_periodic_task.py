# base_app/management/commands/remove_periodic_task.py
from django.core.management.base import BaseCommand
from django_celery_beat.models import PeriodicTask

class Command(BaseCommand):
    help = 'Remove periodic task'

    def add_arguments(self, parser):
        parser.add_argument('task_name', type=str, help='The name of the task to be removed')

    def handle(self, *args, **kwargs):
        task_name = kwargs['task_name']
        try:
            task = PeriodicTask.objects.get(name=task_name)
            task.delete()
            self.stdout.write(self.style.SUCCESS(f'Successfully removed periodic task: {task_name}'))
        except PeriodicTask.DoesNotExist:
            self.stdout.write(self.style.WARNING(f'No periodic task found with name: {task_name}'))
