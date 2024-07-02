import requests
from django.http import JsonResponse

from coinbase_api.enums import Database
from ..views.views import cb_auth, cb_list_accounts
from ..models.models import AbstractOHLCV, Account, Bitcoin, CryptoMetadata, Ethereum, Polkadot
import json
from ..constants import crypto_models

from datetime import datetime, timedelta
# from django.utils.timezone import make_aware, make_naive
# from django.apps import apps
from ..enums import Method, Granularities
# from .utilities.utils import cb_fetch_product_candles
from .ml_utils import add_calculated_parameters
import json




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
        data = cb_auth.restClientInstance.get_products()
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
        data = cb_auth.restClientInstance.get_candles(product_id=product_id, start=start, end=end, granularity=granularity)
        if 'errors' in data:
            # print(data['errors'])  # Logging the error for debugging purposes
            return JsonResponse({'errors': data['errors']}, status=data.get('status', 500))
        # If you reach here, it means the request was successful
        return JsonResponse(data)
    
    except Exception as e:
        # This will handle any other unforeseen exceptions
        return JsonResponse({'errors': str(e)}, status=500)

def cb_fetch_available_crypto():
    print('trying to fetch crypto...')
    cursor = None
    has_next = True
    while has_next:
        data = cb_list_accounts(cursor)
        response_content = data.content.decode('utf-8')
        json_data = json.loads(response_content)
        accounts_data = json_data.get('accounts', [])
        print(json_data)
        print(accounts_data)
        for account_data in accounts_data:
            # account = Account.objects.get_or_create(name=account_data.get('name', ''), uuid=)
            try:
                account = Account.objects.get(
                    name=account_data.get('name', ''),
                    uuid=account_data.get('uuid', ''),
                )
            except Account.DoesNotExist:
                account = Account(
                    name=account_data.get('name', ''),
                    uuid=account_data.get('uuid', ''),
                )

            # account, created = Account.objects.get_or_create(
            #     name=account_data.get('name', ''),
            #     uuid=account_data.get('uuid', ''),
            # )
            account.currency=account_data.get('currency', ''),
            account.value=account_data.get('available_balance', {}).get('value', 0)
            account.save()

        has_next = data.get('has_next', False)
        cursor = data.get('cursor')
        print(f'Fetching of crypto done. did store {len(accounts_data)} items.')
        # TODO
        #! the break is here since the cb_auth method currently does not support pagination
        #! since a header token is needed for the pagination requests. 
        #! see https://docs.cloud.coinbase.com/exchange/docs/rest-pagination for more details
        break

def initialize_default_cryptos(initial_volume=1000, database=Database.DEFAULT.value):
    all_accounts = Account.objects.using(database).all()
    all_accounts.delete()
    cryptos = crypto_models
    uuid='00000000-0000-0000-0000-000000000000'
    usdc = Account(
        name='USDC Wallet',
        uuid=uuid,
        currency='USDC',
        value=initial_volume,
    )
    usdc.save(using=database)
    for crypto in cryptos:
        crypto_account = Account(
            name=f'{crypto.symbol} Wallet',
            uuid=uuid,
            currency=crypto.symbol,
            value=0,
        )
        crypto_account.save(using=database)

def calculate_total_volume(database=Database.DEFAULT.value, base_currency_name='USDC'):
    cryptos = crypto_models
    try:
        base_currency = Account.objects.using(database).get(name=f'{base_currency_name} Wallet')
    except Account.DoesNotExist:
        print(f'{base_currency_name} does not exist')
        return
    volume = base_currency.value
    # print(f'{base_currency_name} value: {volume}')
    for crypto in cryptos:
        try:
            latest_entry = crypto.objects.using(database).latest('timestamp')
        except crypto.DoesNotExist:
            print(f'Could not find: {crypto.symbol} data')
            continue
        try:
            crypt = Account.objects.using(database).get(name=f'{crypto.symbol} Wallet')
        except Account.DoesNotExist:
            print(f'Could not find: {crypto.symbol} Wallet')
            continue
        volume += (crypt.value * latest_entry.close)
    return volume


########################   Here starts the historical db update   ######################

def last_full_hour(date_time):
    return date_time.replace(minute=0, second=0, microsecond=0)

def update_ohlcv_data():
    cryptos = [Bitcoin, Ethereum, Polkadot]  # Models for cryptos we want to update
    granularity = 3600  # 1 hour in seconds

    for crypto in cryptos:
        print(f'Starting with crypto: {crypto.__name__}')
        #! delete the following two lines, only keep if need to redo database arises
        # all_data = crypto.objects.all()
        # all_data.delete()

        latest_entry = crypto.objects.order_by('-timestamp').first()
        # start = latest_entry.timestamp if latest_entry else last_full_hour(datetime.utcnow()) - timedelta(days=30)
        start = latest_entry.timestamp if latest_entry else last_full_hour(datetime.now()) - timedelta(days=30)
        end = last_full_hour(datetime.now())
        delta = end - start
        required_data_points = delta.total_seconds() / granularity
        chunks = -(-required_data_points // 300)  # Calculate the number of chunks (ceiling division)
        for _ in range(int(chunks)):
            print(f'{datetime.now()}: starting with chunk {_+1} of {int(chunks)}')
            tmp_end = start + timedelta(hours=300)
            print(f'requesting data for {start.year}.{start.month}.{start.day}:{start.hour}-{tmp_end.year}.{tmp_end.month}.{tmp_end.day}:{tmp_end.hour}')
            data = cb_fetch_product_candles(f'{crypto.symbol}-USD', int(datetime.timestamp(start)), int(datetime.timestamp(tmp_end)), Granularities.ONE_HOUR.value)
            json_data = json.loads(data.content)
            store_data(crypto, json_data["candles"])
            start += timedelta(hours=300)  # Move start ahead by 300 hours
            print(f'{datetime.now()}: finished with chunk {_+1} of {int(chunks)}')

        # Deleting data older than a month
        add_calculated_parameters(crypto)
        month_ago = datetime.utcnow() - timedelta(days=30)
        crypto.objects.filter(timestamp__lt=month_ago).delete()
        print(f'Finished with crypto: {crypto.__name__}, {crypto._meta.model_name}')
    #! need to update new_data to fetch subset of data and also fetch all crypto data types
    # new_data = Bitcoin.objects.all()
    # chain(predict_with_lstm.s(new_data), predict_with_xgboost.s(new_data)).apply_async()

def store_data(crypto_model, data, database=Database.DEFAULT.value):
    try:
        entries = []
        for item in data:
            try:
                timestamp = float(item["start"])
                low = float(item["low"])
                high = float(item["high"])
                opening = float(item["open"])
                close_base = float(item["close"])
                volume = float(item["volume"])
                
                new_entry = crypto_model(
                    timestamp=datetime.utcfromtimestamp(int(timestamp)),
                    open=opening,
                    high=high,
                    low=low,
                    close=close_base,
                    volume=volume,
                )
                entries.append(new_entry)
            except Exception as e:
                print(f'Encountered an error with item {item}: {e}')
        
        # Use bulk_create to insert all entries at once
        crypto_model.objects.using(database).bulk_create(entries)
    except Exception as e:
        print(f'Encountered the following error: {e}')

def cb_find_earliest_data(product_id='BTC-USDC'):
    print(f'product_id: {product_id}')
    granularity = Granularities.ONE_DAY.value
    end = int(datetime.timestamp(datetime.now()))
    earliest_date = None
    i = 0
    data = None
    while True:
        end = int(datetime.timestamp(datetime.now() - timedelta(days=300*i)))
        start = int(datetime.timestamp(datetime.now() - timedelta(days=300*(i+1))))
        try:
            data = cb_fetch_product_candles(product_id, start, end, granularity)
            tmp = json.loads(data.content)
            # print(f'loaded data: {tmp}')
        except TypeError:
            print(f'cb_find_earliest_data: data could not be serialized to json. Data: {data}')
            break
        try:
            print(f'for {product_id} found the following data for idx {i}: {len(tmp["candles"])} \nLast timestamp was: {tmp["candles"][-1]["start"]}')
            earliest_date = int(tmp['candles'][-1]['start'])
        except (IndexError, KeyError) as error:
            print(f'Error trying to get the earliest date: {error}')
            break
        data = None
        i+=1

    return earliest_date

def fetch_hourly_data_for_crypto(crypto_model:AbstractOHLCV):
    # symbol = crypto_model.symbol_to_storage(crypto_model.symbol)
    
    # Decide the starting point
    try:
        latest_data = crypto_model.objects.using(Database.HISTORICAL.value).latest('timestamp')
        print(f'found data for {crypto_model.symbol}: {latest_data.timestamp}')
        start = latest_data.timestamp
    except crypto_model.DoesNotExist:
        # If no data is found for this cryptocurrency, use the timestamp from the CryptoMetadata model
        print(f'no data found in db: {crypto_model.symbol}')
        meta_data = CryptoMetadata.objects.using(Database.HISTORICAL.value).get(symbol=CryptoMetadata.symbol_to_storage(crypto_model.symbol))
        start = meta_data.earliest_date

    # start_time_unix = int(datetime.timestamp(start_time))
    # end_time_unix = int(datetime.timestamp(datetime.now()))
    end = last_full_hour(datetime.now())
    delta = end - start
    required_data_points = delta.total_seconds() / 3600
    chunks = -(-required_data_points // 300)  # Calculate the number of chunks (ceiling division)
    if (chunks < 1): #only update if there is a lot of data to update. makes the whole process MUCH faster
        return
    for _ in range(int(chunks)):
        # Fetch hourly data
        print(f'starting with chunk {_+1} of {int(chunks)}')
        tmp_end = start + timedelta(hours=300)
        print(f'requesting data for {start.year}.{"0" if start.month < 10 else ""}{start.month}.{"0" if start.day < 10 else ""}{start.day}-{"0" if start.hour < 10 else ""}{start.hour}:00-{tmp_end.year}.{"0" if tmp_end.month < 10 else ""}{tmp_end.month}.{"0" if tmp_end.day < 10 else ""}{tmp_end.day}-{"0" if tmp_end.hour < 10 else ""}{tmp_end.hour}:00')
        data = cb_fetch_product_candles(f'{crypto_model.symbol}-USDC', int(datetime.timestamp(start)), int(datetime.timestamp(tmp_end)), Granularities.ONE_HOUR.value)
        if 'errors' in json.loads(data.content):
            print(f'Could not find data for chunk: {_+1}/{int(chunks)}!')
            print('Aborting to keep database integrity intact!')
            break
        json_data = json.loads(data.content)
        store_data(crypto_model, json_data["candles"], 'historical')
        start += timedelta(hours=300)  # Move start ahead by 300 hours
        print(f'finished with chunk {_+1} of {int(chunks)}')
        
    add_calculated_parameters(crypto_model, database=Database.HISTORICAL.value)


def healthcheck_data(crypto_model):
    missing_data = []

    try:
        earliest_data = crypto_model.objects.earliest('timestamp')
        start_time = earliest_data.timestamp
    except crypto_model.DoesNotExist:
        # If no data is found for this cryptocurrency, use the timestamp from the CryptoMetadata model
        meta_data = CryptoMetadata.objects.get(symbol=crypto_model.symbol)
        start_time = meta_data.earliest_date

    current_time = start_time
    end_time = datetime.now()

    while current_time <= end_time:
        try:
            crypto_model.objects.get(timestamp=current_time)
        except crypto_model.DoesNotExist:
            missing_data.append(current_time)

        # Move to the next hour
        current_time += timedelta(hours=1)

    return missing_data