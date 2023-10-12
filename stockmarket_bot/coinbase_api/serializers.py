# serializers.py

from rest_framework import serializers
from .models import Cryptocurrency, Bitcoin, Ethereum, Polkadot

class CryptocurrencySerializer(serializers.ModelSerializer):
    class Meta:
        model = Cryptocurrency
        fields = ['id', 'base_display_symbol', 'quote_display_symbol','product_id', 'trading_indicator']

from rest_framework import serializers

crypto_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# class AbstractOHLCVSerializer(serializers.ModelSerializer):
#     class Meta:
#         fields = crypto_fields

class BitcoinSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bitcoin
        fields=crypto_fields

class EthereumSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ethereum
        fields=crypto_fields

class PolkadotSerializer(serializers.ModelSerializer):
    class Meta:
        model = Polkadot
        fields=crypto_fields
