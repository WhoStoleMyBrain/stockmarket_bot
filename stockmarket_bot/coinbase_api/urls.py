# urls.py
from coinbase_api.views.views import (
    cb_fetch_best_bid_ask, 
    cb_fetch_best_bid_asks, 
    cb_fetch_coinbase_data, 
    cb_fetch_coinbase_data_differently, 
    cb_fetch_currencies, 
    cb_fetch_data_for_btc,
    cb_list_orders,
    cb_fetch_exchange_rates,
    cb_fetch_product_book,
    cb_get_account,
    cb_get_market_trades,
    cb_list_accounts,
    bitcoin_chart,
    command_buttons,
    )

from coinbase_api.api_views.api_views import (
    BitcoinIncompleteView,
    BitcoinView,
    CryptoMetadataView,
    CryptocurrencyViewSet,
    EthereumIncompleteView,
    EthereumView,
    PolkadotHistoricalView,
    PolkadotIncompleteView,
    PolkadotView,
    PredictionView,
    BitcoinData,
    AccountViewSet,
    cb_fetch_product_view,
    cb_fetch_product_candles_view, 
    cb_fetch_product_list_view,
    write_currencies_to_file,
    # cb_fetch_products_view
)

from django.urls import path, include
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'bitcoin', BitcoinView)
router.register(r'ethereum', EthereumView)
router.register(r'polkadot', PolkadotView)
router.register(r'polkadothistorical', PolkadotHistoricalView)
router.register(r'prediction', PredictionView)
router.register(r'accounts', AccountViewSet)
router.register(r'cryptocurrencies', CryptocurrencyViewSet, basename='cryptocurrency')
router.register(r'crypto-metadata', CryptoMetadataView)
# router.register(r'incomplete_bitcoin', BitcoinIncompleteView, basename='def1')
# router.register(r'incomplete_ethereum', EthereumIncompleteView, basename='def2')
# router.register(r'incomplete_polkadot', PolkadotIncompleteView, basename='def3')
# router.register(r'bitcoin-data', BitcoinData)

urlpatterns = [
    path('cb-fetch-coinbase-data/', cb_fetch_coinbase_data, name='cb_fetch_coinbase_data'),
    path('cb-fetch-coinbase-data-v2/', cb_fetch_coinbase_data_differently, name='cb_fetch_coinbase_data_v2'),
    path('cb-fetch-currencies/', cb_fetch_currencies, name='cb_fetch_currencies'),
    path('cb-fetch-exchange-rates/', cb_fetch_exchange_rates, name='cb_fetch_exchange_rates'),
    path('cb-fetch-data-for-btc/', cb_fetch_data_for_btc, name='cb_fetch_data_for_btc'),
    path('cb-fetch-best-bid-asks/', cb_fetch_best_bid_asks, name='cb_fetch_best_bid_asks'),
    path('cb-fetch-best-bid-ask/', cb_fetch_best_bid_ask, name='cb_fetch_best_bid_ask'),
    path('cb-fetch-product-book/', cb_fetch_product_book, name='cb_fetch_product_book'),
    path('cb-fetch-product-list/', cb_fetch_product_list_view, name='cb_fetch_product_list'),
    path('write-currencies-to-file/', write_currencies_to_file, name='write_currencies_to_file'),
    path('cb-fetch-product/', cb_fetch_product_view, name='cb_fetch_product'),
    # path('cb-fetch-products/', cb_fetch_products_view, name='cb_fetch_products'),
    path('cb-list-accounts/', cb_list_accounts, name='cb_list_accounts'),
    path('cb-get-account/', cb_get_account, name='cb_get_account'),
    path('cb-fetch-product-candles/', cb_fetch_product_candles_view, name='cb_fetch_product_candles'),
    path('cb-get-market-trades/', cb_get_market_trades, name='cb_get_market_trades'),
    path('cb-list-orders/', cb_list_orders, name='cb_list_orders'),
    path('bitcoins/chart/', bitcoin_chart, name='bitcoin-chart'),
    path('api/incomplete_bitcoin/', BitcoinIncompleteView.as_view()),
    path('api/incomplete_ethereum/', EthereumIncompleteView.as_view()),
    path('api/incomplete_polkadot/', PolkadotIncompleteView.as_view()),
    path('api/bitcoin/', BitcoinData.as_view(), name='bitcoin-data'),
    path('api/', include(router.urls)),
    path('commands/', command_buttons, name='command_buttons'),
]
