# serializers.py

from rest_framework import serializers
from .models.models import Account, Cryptocurrency, Bitcoin, Ethereum, Polkadot, Prediction

class CryptocurrencySerializer(serializers.ModelSerializer):
    class Meta:
        model = Cryptocurrency
        fields = ['id', 'base_display_symbol', 'quote_display_symbol','product_id', 'trading_indicator']



crypto_fields = [
    'timestamp', 
    'close', 
    'volume', 
    'sma', 
    'ema', 
    'rsi', 
    'macd', 
    'bollinger_high', 
    'bollinger_low', 
    'vmap',
    'percentage_returns',
    'log_returns',
    ]

extra_kwargs = {
    'sma': {'required': False},
    'ema': {'required': False},
    'rsi': {'required': False},
    'macd': {'required': False},
    'bollinger_high': {'required': False},
    'bollinger_low': {'required': False},
    'vmap': {'required': False},
    'percentage_returns': {'required': False},
    'log_returns': {'required': False},
    } 

class BitcoinSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bitcoin
        fields=crypto_fields
        extra_kwargs = extra_kwargs

class BitcoinViewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bitcoin
        fields = ['timestamp', 'close']

class EthereumSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ethereum
        fields=crypto_fields
        extra_kwargs = extra_kwargs

class PolkadotSerializer(serializers.ModelSerializer):
    class Meta:
        model = Polkadot
        fields=crypto_fields
        extra_kwargs = extra_kwargs

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'

class AccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = Account
        fields = ['name', 'uuid', 'currency', 'value']

# import numpy as np
# from django.core.serializers.json import DjangoJSONEncoder

# class NaNJSONEncoder(DjangoJSONEncoder):
#     def default(self, obj):
#         print(f'obj: {obj}')
#         if isinstance(obj, float) and np.isnan(obj):
#             return None  # or use the string "NaN"
#         return super().default(obj)
