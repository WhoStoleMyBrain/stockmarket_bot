from celery import shared_task

# from coinbase_api.utilities.cb_provider import CbProvider
# from coinbase_api.utilities.rl_action_handler import RlActionHandler
# from coinbase_api.utilities.rl_provider import RlProvider
from coinbase_api.providers import cb_provider, rl_provider, rl_action_handler

@shared_task
def print_statement():
    print('Hello, World!')

@shared_task
def run_trading_bot():
    # cb_provider = CbProvider()
    # rl_provider = RlProvider()
    # rl_action_handler = RlActionHandler()

    cb_provider.update()
    all_actions = rl_provider.get_all_actions()
    rl_action_handler.handle_actions(all_actions)
        