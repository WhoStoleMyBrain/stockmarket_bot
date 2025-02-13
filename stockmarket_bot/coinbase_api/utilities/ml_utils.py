
from collections import defaultdict
from django.utils.timezone import make_aware
import numpy as np
import pandas as pd
from coinbase_api.models.models import AbstractOHLCV
import ta
from django.db.models import Sum, F, FloatField, ExpressionWrapper
from ..enums import Database
from typing import Union, List
from django.db.models import QuerySet

def calculate_vmap(data: Union[QuerySet, List[AbstractOHLCV]]):
    if hasattr(data, 'aggregate'):
        aggregated_data = data.aggregate(
            total_volume=Sum('volume'),
            total_value=Sum(
                ExpressionWrapper(
                    ((F('open') + F('close') + F('high')) / 3) * F('volume'),
                    output_field=FloatField()
                )
            )
        )
        if aggregated_data['total_volume'] != 0:
            vmap = aggregated_data['total_value'] / aggregated_data['total_volume']
        else:
            vmap = 0
        return vmap
    else:
        total_volume = 0.0
        total_value = 0.0
        for item in data:
            # Ensure that volume is available; skip items with no volume
            if item.volume is None:
                continue
            volume = item.volume
            # Use 0 as a fallback if any price is missing (you may adjust this logic)
            open_price = item.open if item.open is not None else 0.0
            close_price = item.close if item.close is not None else 0.0
            high_price = item.high if item.high is not None else 0.0
            # Compute the average price for this item
            price = (open_price + close_price + high_price) / 3
            total_volume += volume
            total_value += price * volume

        return total_value / total_volume if total_volume != 0 else 0

def calculate_percentage_returns(current_item, previous_item):
    if previous_item is None:
        return None
    if previous_item.close == 0:
        return 0.0
    else:
        return (current_item.close - previous_item.close) / (previous_item.close)
    
def calculate_log_returns(current_item, previous_item):
    if previous_item is None:
        return None
    if abs(current_item.close - previous_item.close) < 1e-15:
        return 0.0
    else:
        # print(f'current item: {current_item.close}, previous_item: {previous_item.close}')
        return np.log(current_item.close / previous_item.close)
    
def calculate_shifted_value(item, last_168_data, index):
    try:
        return last_168_data[index].close > item.close
    except IndexError:
        return False

def check_nan(value):
    return value if not pd.isna(value) else None

def add_calculated_parameters(crypto_model: AbstractOHLCV, database=Database.DEFAULT.value):
    sequence_length = 50
    rsi_length = 14
    bollinger_length = 20
    batch_size = 500  # Define a batch size for bulk updates
    # -------------------------------------------------------------------------
    # 1. Preload all data into memory, sorted by timestamp (ascending)
    #    This avoids multiple DB hits when iterating.
    # -------------------------------------------------------------------------
    all_data = list(crypto_model.objects.using(database).order_by('timestamp').all())
    total_length = len(all_data)
    
    day_data = defaultdict(list)
    for item in all_data:
        day_data[item.timestamp.date()].append(item)
    # Sort each day’s items in descending order to mimic order_by('-timestamp')
    for day, items in day_data.items():
        items.sort(key=lambda x: x.timestamp, reverse=True)
    
     # -------------------------------------------------------------------------
    # 3. Precompute last_168_data once.
    #    This query does not depend on each item and thus can be computed outside.
    # -------------------------------------------------------------------------
    # hours_ago = make_aware(datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(hours=168))
    # last_168_data = [item for item in all_data if item.timestamp >= hours_ago]
    
    close_values = [item.close for item in all_data]

    items_to_update = []
    previous_item = None

    # Iterate through all data
    for idx, item in enumerate(all_data):
        
        if (idx + 1) % 2500 == 0:
            print(f'{crypto_model.symbol}: Processing item {idx + 1}/{total_length}')
        
        # Skip items that are already updated
        if item.all_fields_set():
            previous_item = item
            continue

        # Retrieve data for the day using the precomputed grouping
        day = item.timestamp.date()
        data_day_of_item = day_data.get(day, [])

        # Fetch data needed for calculations
        # data_day_of_item = crypto_model.objects.using(database).filter(
        #     timestamp__gte=item.timestamp.replace(hour=0, minute=0, second=0),
        #     timestamp__lte=item.timestamp.replace(hour=23, minute=59, second=59)
        # ).order_by('-timestamp')

        # hours_ago = make_aware(datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(hours=168))
        # last_168_data = crypto_model.objects.using(database).filter(
        #     timestamp__gte=hours_ago
        # ).order_by('timestamp')

        # data_sequence_length = crypto_model.objects.using(database).filter(
        #     timestamp__lte=item.timestamp
        # ).order_by('-timestamp')[:sequence_length]

        # close = list(data_sequence_length.values_list('close', flat=True))
        # close.append(item.close)
        
        # Obtain the sliding window of closing prices
        start_idx = max(0, idx - sequence_length + 1)
        sequence_closes = close_values[start_idx: idx + 1]
        close_series = pd.Series(sequence_closes)
        
        # close = pd.Series(close)
        close_bollinger = close_series[-bollinger_length:]
        close_rsi = close_series[-rsi_length:]

        # Calculate parameters
        item.vmap = calculate_vmap(data_day_of_item)
        item.percentage_returns = calculate_percentage_returns(item, previous_item)
        item.log_returns = calculate_log_returns(item, previous_item)
        # item.close_higher_shifted_1h = calculate_shifted_value(item, last_168_data, 1)
        # item.close_higher_shifted_24h = calculate_shifted_value(item, last_168_data, 24)
        # item.close_higher_shifted_168h = calculate_shifted_value(item, last_168_data, 168)
        item.sma = check_nan(ta.trend.sma_indicator(close_series, sequence_length).iloc[-1])
        item.ema = check_nan(ta.trend.ema_indicator(close_series, sequence_length).iloc[-1])
        item.macd = check_nan(
            ta.trend.ema_indicator(close_series, 12).iloc[-1] - ta.trend.ema_indicator(close_series, 26).iloc[-1]
        )
        bollinger = ta.volatility.BollingerBands(close_bollinger, bollinger_length, 2)
        item.bollinger_high = check_nan(bollinger.bollinger_hband().iloc[-1])
        item.bollinger_low = check_nan(bollinger.bollinger_lband().iloc[-1])
        item.rsi = check_nan(ta.momentum.rsi(close_rsi, rsi_length).iloc[-1])

        # Add the item to the list of items to update
        items_to_update.append(item)
        previous_item = item

        # Update items in batches
        if len(items_to_update) >= batch_size:
            crypto_model.objects.using(database).bulk_update(items_to_update, [
                'vmap', 'percentage_returns', 'log_returns',
                # 'close_higher_shifted_1h', 'close_higher_shifted_24h', 'close_higher_shifted_168h',
                'sma', 'ema', 'macd', 'bollinger_high', 'bollinger_low', 'rsi'
            ])
            items_to_update = []

    # Final batch update for remaining items
    if items_to_update:
        crypto_model.objects.using(database).bulk_update(items_to_update, [
            'vmap', 'percentage_returns', 'log_returns',
            # 'close_higher_shifted_1h', 'close_higher_shifted_24h', 'close_higher_shifted_168h',
            'sma', 'ema', 'macd', 'bollinger_high', 'bollinger_low', 'rsi'
        ])
