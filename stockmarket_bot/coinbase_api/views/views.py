import requests
from django.http import JsonResponse
from coinbase.wallet.client import Client
from constants import API_KEY, API_SECRET, crypto_models
import datetime
from ..cb_auth import CBAuth, Method, OrderStatus
from ..models.models import Bitcoin, Prediction
from django.shortcuts import render
from ..enums import Method, OrderStatus

cb_auth = CBAuth()
cb_auth.set_credentials(API_KEY, API_SECRET)

def cb_fetch_coinbase_data(request):
    try:
        response = requests.get('https://api.coinbase.com/v2/prices/spot', params={'currency': 'USD'})
        response.raise_for_status()  # Check if the request was successful
        data = response.json()
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)

def cb_fetch_coinbase_data_differently(request):
    try:
        client = Client(API_KEY, API_SECRET)
        price = client.get_buy_price(currency_pair = 'BTC-USD')
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(price)

def cb_fetch_currencies(request):
    try:
        client = Client(API_KEY, API_SECRET)
        currencies = client.get_currencies()
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(currencies)
    
def cb_fetch_exchange_rates(request):
    try:
        client = Client(API_KEY, API_SECRET)
        exchange_rates = client.get_exchange_rates()
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(exchange_rates)

def cb_fetch_data_for_btc(request):
    try:
        client = Client(API_KEY, API_SECRET)
        buy_price = client.get_buy_price(currency_pair = 'BTC-EUR')
        sell_price = client.get_sell_price(currency_pair = 'BTC-EUR')
        spot_price = client.get_spot_price(currency_pair = 'BTC-EUR')
        all_prices = {
            'buy_price': buy_price,
            'sell_price': sell_price,
            'spot_price': spot_price,
        }
        
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(all_prices)
    
def cb_fetch_best_bid_asks(request):
    """
    Retrieves the best bid/ask price and size for all items traded on cb\n
    Datastructure:\n
    pricebooks\n
    --[id]\n
    ----product_id\n
    ----bids\n
    ------[id]\n
    --------price\n
    --------size\n
    ----asks\n
    ------[id]\n
    --------price\n
    --------size\n
    ----time\n
    """
    try:
        data = cb_auth(Method.GET.value, "/api/v3/brokerage/best_bid_ask")
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)

def cb_fetch_best_bid_ask(request):
    """
    Retrieves the best bid/ask price and size for single item traded on cb\n
    Datastructure:\n
    pricebooks\n
    --[id]\n
    ----product_id\n
    ----bids\n
    ------[id]\n
    --------price\n
    --------size\n
    ----asks\n
    ------[id]\n
    --------price\n
    --------size\n
    ----time\n
    """
    try:
        product_ids = ['BTC-USD', 'ETH-USD']
        data = cb_auth(Method.GET.value, f"/api/v3/brokerage/best_bid_ask?{'&'.join([f'product_ids={id}' for id in product_ids])}")
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)

def cb_fetch_product_book(request):
    """
    Returns current asks and bids for the given product with product_id. Returns 'limit' items\n
    Datastructure:\n
    pricebooks\n
    --product_id\n
    --bids\n
    ----[id]\n
    ------price\n
    ------size\n
    --asks\n
    ----[id]\n
    ------price\n
    ------size\n
    """
    try:
        params = {
            'product_id': 'BTC-USD',
            'limit': 32,
        }
        data = cb_auth(Method.GET.value, "/api/v3/brokerage/product_book", '', params)
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)


def cb_fetch_product(product_id):
    """
    @param product_ids - List of product IDs to return.\n
    Get information on a single product by product ID.\n
    datastructure: \n
    -product_id\n
    -price\n
    -price_percentage_change_24h\n
    -volume_24h\n
    -volume_percentage_change_24h\n
    -base_increment\n
    -quote_increment\n
    -quote_min_size\n
    -quote_max_size\n
    -base_min_size\n
    -base_max_size\n
    -base_name\n
    -quote_name\n
    -watched\n
    ... for more see result\n
    """
    try:
        data = cb_auth(Method.GET.value, f"/api/v3/brokerage/products/{product_id}")
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    except KeyError as e:
        return JsonResponse({'error': f'KeyError: {e}'}, status=500)
    return JsonResponse(data)

def cb_fetch_products():
    """
    @param product_ids - List of product IDs to return.\n
    Get information on a single product by product ID.\n
    datastructure: \n
    -product_id\n
    -price\n
    -price_percentage_change_24h\n
    -volume_24h\n
    -volume_percentage_change_24h\n
    -base_increment\n
    -quote_increment\n
    -quote_min_size\n
    -quote_max_size\n
    -base_min_size\n
    -base_max_size\n
    -base_name\n
    -quote_name\n
    -watched\n
    ... for more see result\n
    """
    try:
        data = cb_auth(Method.GET.value, f"/api/v3/brokerage/products")
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    except KeyError as e:
        return JsonResponse({'error': f'KeyError: {e}'}, status=500)
    return JsonResponse(data)

def cb_list_accounts(request):
    '''
    @param limit - the pagination limit\n
    @param cursor - cursor used for pagination (basically offset)\n
    Even though this endpoints tells you accounts, it is actually all the wallets
    (e.g. BTC, ETH, DOT) that are linked to your account
    -> This is the endpoint where the available assets can be retrieved
    -> Note: It seems that staked assets are not available here, as intended
    datastructure:
    accounts
    --[id]
    ----uuid
    ----name
    ----currency
    ----available_balance
    ------value
    ------currency
    ----ready
    ----hold
    ------value
    ------currency
    has_next
    cursor
    size
    '''
    try:
        data = cb_auth(Method.GET.value, "/api/v3/brokerage/accounts")
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)

def cb_get_account(request):
    '''
    @param account_uuid - UUID of the account to be retrieved\n
    Even though this endpoints tells you account, it is actually single wallets
    (e.g. BTC, ETH, DOT) that are linked to your account
    Retrieves information about a single account, however same info as for the 
    list_accounts endpoint is returned
    '''
    try:
        account_uuid = '95f486aa-ac65-51b9-9da8-af08796f89ac'#! only example uuid
        data = cb_auth(Method.GET.value, f"/api/v3/brokerage/accounts/{account_uuid}")
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)

def cb_get_market_trades(request):
    '''
    @param product_id - The trading pair, i.e., 'BTC-USD'\n
    @query_param limit - Number of trades to return.\n
    Get snapshot information, by product ID, about the last trades (ticks), 
    best bid/ask, and 24h volume.
    datastructure:\n
    trades\n
    --[id]\n
    ----trade_id\n
    ----product_id\n
    ----price\n
    ----size\n
    ----time\n
    ----side\n
    ----bid\n
    ----ask\n
    best_bid\n
    best_ask\n
    '''
    try:
        product_id = 'BTC-USD'
        params = {"limit":30}
        data = cb_auth(Method.GET.value, f"/api/v3/brokerage/products/{product_id}/ticker", "", params)
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)

def cb_list_orders(request):
    '''
    Get a list of orders filtered by optional query parameters (product_id, order_status, etc).
    @query_param product_id - Optional string of the product ID. Defaults to null, or fetch for all products.
    @query_param order_status - A list of order statuses.
    @query_param limit - A pagination limit with no default set. If has_next is true, 
        	            additional orders are available to be fetched with pagination; also the cursor value in the response can be passed as cursor parameter in the subsequent request.
    @query_param start_date - Start date to fetch orders from, inclusive.
    @query_param end_date - An optional end date for the query window, exclusive. If provided only orders with creation time before this date will be returned.
    @query_param order_type - Type of orders to return. Default is to return all order types.
    @query_param order_side - Only orders matching this side are returned. Default is to return all sides.
    @query_param cursor - Cursor used for pagination. When provided, the response returns responses after this cursor.
    @query_param product_type - Only orders matching this product type are returned. Default is to return all product types.
    @query_param order_placement_source - Only orders matching this placement source are returned. Default is to return RETAIL_ADVANCED placement source.
    @query_param contract_expiry_type - Only orders matching this contract expiry type are returned. Filter is only applied if ProductType is set to FUTURE in the request.
    '''
    try:
        params = {
            "order_status": OrderStatus.OPEN.value,
            "start_date": datetime.datetime(2023, 10, 1).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        data = cb_auth(Method.GET.value, f"/api/v3/brokerage/orders/historical/batch", "", params)
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)

def access_prediction_data(request):
    from datetime import datetime, timedelta

    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()

    # Fetch actual data (assuming your data model is named 'ActualData')
    actual_data = Bitcoin.objects.filter(timestamp__range=(start_time, end_time))

    # Fetch predicted data for LSTM
    lstm_predictions = Prediction.objects.filter(
        model_name='LSTM',
        timestamp_predicted_for__range=(start_time, end_time)
    )

    # Similarly for other models...

def bitcoin_chart(request):
    return render(request, 'chart.html')