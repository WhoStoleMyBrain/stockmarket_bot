from django.core.management.base import BaseCommand
# from coinbase_api.utilities.cb_provider import CbProvider
# from coinbase_api.utilities.rl_action_handler import RlActionHandler
# from coinbase_api.utilities.rl_provider import RlProvider

# cb_provider = CbProvider()
# rl_provider = RlProvider()
# rl_action_handler = RlActionHandler()
from coinbase_api.providers import cb_provider, rl_provider, rl_action_handler

class Command(BaseCommand):
    help = 'Test the real environment WITHOUT actually modifying the state'

    def handle(self, *args, **kwargs):
        # cb_provider = CbProvider()
        # rl_provider = RlProvider()
        # rl_action_handler = RlActionHandler()

        cb_provider.update()
        all_actions = rl_provider.get_all_actions()
        rl_action_handler.handle_actions(all_actions)
        
    