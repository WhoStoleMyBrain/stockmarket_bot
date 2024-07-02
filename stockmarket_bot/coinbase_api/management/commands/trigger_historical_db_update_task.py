from django.core.management.base import BaseCommand
from coinbase_api.models.generated_models import *
from coinbase_api.utilities.utils import fetch_hourly_data_for_crypto
from coinbase_api.constants import crypto_models

class Command(BaseCommand):
    help = 'Trigger Historical Database Update Task'

    def handle(self, *args, **kwargs):
        print('starting historical db update task')
        for crypto_model in crypto_models:
            # if (crypto_model in [USDT, ZRX, AMP, EOS, MASK, MPL, PRO, DAI, FIDA, XCN, BADGER, AERGO, GST, RBN, STORJ, OXT, CGLD, AURORA, GTC, CLV, ARPA, DNT, FOX, AVT, XYO, FORTH, WAMPL, INDEX, BTRST, HOPR, DIA, LQTY, RARI, PREFIX00, RAD, BNT, WCFG, ERN, BAL, CTX, PLU, MUSE, PYUSD, TIME, FX, BIT, DEXT, LOKA, PAX, DAR, ELA, GUSD, GYEN, RAI]):
            fetch_hourly_data_for_crypto(crypto_model)
        