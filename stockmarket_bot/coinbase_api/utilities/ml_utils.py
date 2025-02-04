
from datetime import datetime, timedelta, timezone
from django.utils.timezone import make_aware
import numpy as np
import pandas as pd
from coinbase_api.models.models import AbstractOHLCV
import ta
from django.db.models import Sum, F, FloatField, ExpressionWrapper
from ..enums import Database
from typing import Union, List
from django.db.models import QuerySet
import pandas as pd
from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django.db.models import Q
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

def calculate_vmap(queryset: Union[QuerySet, List[AbstractOHLCV]]):
    aggregated_data = queryset.aggregate(
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

def calculate_percentage_returns(current_item, previous_item):
    if previous_item is None:
        return None
    if previous_item.close == 0:
        return None
    else:
        return (current_item.close - previous_item.close) / (previous_item.close)
    
def calculate_log_returns(current_item, previous_item):
    if previous_item is None:
        return None
    if abs(current_item.close - previous_item.close) < 1e-15:
        return None
    else:
        # print(f'current item: {current_item.close}, previous_item: {previous_item.close}')
        return np.log(current_item.close / previous_item.close)
    
def calculate_shifted_value(item, last_168_data, index):
    try:
        return last_168_data[index].close > item.close
    except IndexError:
        return None

def check_nan(value):
    return value if not pd.isna(value) else None

def add_calculated_parameters(crypto_model: AbstractOHLCV, database=Database.DEFAULT.value):
    sequence_length = 50
    rsi_length = 14
    bollinger_length = 20
    batch_size = 500  # Define a batch size for bulk updates
    all_data = crypto_model.objects.using(database).all()
    previous_item = None
    idx = 0
    length = len(all_data)

    items_to_update = []

    # Iterate through all data
    for item in all_data:
        idx += 1
        if idx % 2500 == 0:
            print(f'Processing item {idx}/{length}')
        
        # Skip items where all fields are already set
        if item.all_fields_set():
            continue
        
        # Fetch data needed for calculations
        data_day_of_item = crypto_model.objects.using(database).filter(
            timestamp__gte=item.timestamp.replace(hour=0, minute=0, second=0),
            timestamp__lte=item.timestamp.replace(hour=23, minute=59, second=59)
        ).order_by('-timestamp')

        hours_ago = make_aware(datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(hours=168))
        last_168_data = crypto_model.objects.using(database).filter(
            timestamp__gte=hours_ago
        ).order_by('timestamp')

        data_sequence_length = crypto_model.objects.using(database).filter(
            timestamp__lte=item.timestamp
        ).order_by('-timestamp')[:sequence_length]

        close = list(data_sequence_length.values_list('close', flat=True))
        close.append(item.close)
        close = pd.Series(close)
        close_bollinger = close[-bollinger_length:]
        close_rsi = close[-rsi_length:]

        # Calculate parameters
        item.vmap = calculate_vmap(data_day_of_item)
        item.percentage_returns = calculate_percentage_returns(item, previous_item)
        item.log_returns = calculate_log_returns(item, previous_item)
        item.close_higher_shifted_1h = calculate_shifted_value(item, last_168_data, 1)
        item.close_higher_shifted_24h = calculate_shifted_value(item, last_168_data, 24)
        item.close_higher_shifted_168h = calculate_shifted_value(item, last_168_data, 168)
        item.sma = check_nan(ta.trend.sma_indicator(close, sequence_length).iloc[-1])
        item.ema = check_nan(ta.trend.ema_indicator(close, sequence_length).iloc[-1])
        item.macd = check_nan(
            ta.trend.ema_indicator(close, 12).iloc[-1] - ta.trend.ema_indicator(close, 26).iloc[-1]
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
                'close_higher_shifted_1h', 'close_higher_shifted_24h', 'close_higher_shifted_168h',
                'sma', 'ema', 'macd', 'bollinger_high', 'bollinger_low', 'rsi'
            ])
            items_to_update = []

    # Final batch update for remaining items
    if items_to_update:
        crypto_model.objects.using(database).bulk_update(items_to_update, [
            'vmap', 'percentage_returns', 'log_returns',
            'close_higher_shifted_1h', 'close_higher_shifted_24h', 'close_higher_shifted_168h',
            'sma', 'ema', 'macd', 'bollinger_high', 'bollinger_low', 'rsi'
        ])
