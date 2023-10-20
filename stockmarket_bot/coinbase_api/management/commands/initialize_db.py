# base_app/management/commands/setup_periodic_task.py
from django.core.management.base import BaseCommand, CommandError
from coinbase_api.utilities.utils import initialize_default_cryptos

class Command(BaseCommand):
    help = 'Setup periodic task'

    def add_arguments(self, parser):
        parser.add_argument(
            '--value', 
            type=int, 
            help='An optional numeric parameter to be passed to initialize_default_cryptos',
            required=False
        )

    def handle(self, *args, **kwargs):
        param = kwargs.get('value', None)
        if param is not None:
            try:
                param = int(param)
            except ValueError:
                raise CommandError('The provided parameter is not a valid number')
            initialize_default_cryptos(param)
        else:
            initialize_default_cryptos()