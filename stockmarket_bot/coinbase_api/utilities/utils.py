import requests
from django.http import JsonResponse
from ..cb_auth import Method
from ..views.views import cb_auth

def cb_fetch_product_list():
    """
    @param limit - A limit describing how many products to return.\n
    @param offset - Number of products to offset before returning.\n
    @param product_type (ENUM: SPOT or FUTURE) - Type of products to return.\n
    @param product_ids - List of product IDs to return.\n
    @param contract_expiry_type - ENUM: UNKNOWN_CONTRACT_EXPIRY_TYPE or EXPIRING\n
    Get a list of the available currency pairs for trading.\n
    datastructure: \n
    products\n
    --[id]\n
    ----product_id\n
    ----price\n
    ----price_percentage_change_24h\n
    ----volume_24h\n
    ----volume_percentage_change_24h\n
    ----base_increment\n
    ----quote_increment\n
    ----quote_min_size\n
    ----quote_max_size\n
    ----base_min_size\n
    ----base_max_size\n
    ----base_name\n
    ----quote_name\n
    ----watched\n
    ... for more see result\n
    num_products
    """
    try:
        data = cb_auth(Method.GET.value, "/api/v3/brokerage/products")
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)

def cb_fetch_product_candles(product_id, start, end, granularity):
    """
    @param product_ids - List of product IDs to return.\n
    @query_param start - Timestamp for starting range of aggregations, in UNIX time.
    @query_param end - Timestamp for ending range of aggregations, in UNIX time.
    @query_param granularity - The time slice value for each candle. 
    Get rates for a single product by product ID, grouped in buckets.\n
    !MAX 300 datapoints per request!\n
    datastructure: \n
    candles\n
    --[id]\n
    ----start\n
    ----low\n
    ----high\n
    ----open\n
    ----close\n
    ----volume\n
    """
    try:
        params = {
            'start' : start,
            'end' : end,
            'granularity': granularity,
        }
        data = cb_auth(Method.GET.value, f"/api/v3/brokerage/products/{product_id}/candles", '', params)
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)