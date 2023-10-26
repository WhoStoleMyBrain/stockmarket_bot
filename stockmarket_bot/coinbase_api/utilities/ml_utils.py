
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
    
def check_nan(entry):
    '''
    @param entry: any number
    returns None if entry suffices np.isnan, else entry
    '''
    return entry if not np.isnan(entry) else None

def add_calculated_parameters(crypto_model:AbstractOHLCV, database=Database.DEFAULT.value):
    sequence_length = 50
    rsi_length = 14
    bollinger_length = 20
    all_data = crypto_model.objects.using(database).all()
    previous_item = None
    idx = 0
    length = len(all_data)
    #TODO add method to ONLY calculate for entries where any field value is none/null
    for item in all_data:
        idx += 1
        if idx%100==0:
            print(f'processing item {idx}/{length}')
        if item.all_fields_set():
            # print(f'skipping item {idx}/{length} since all values are already set')
            continue
        print(f'ACTUALLY processing item {idx}/{length}')
        data_day_of_item = crypto_model.objects.using(database).filter(timestamp__gte=item.timestamp.replace(hour=0, minute=0, second=0), timestamp__lte=item.timestamp.replace(hour=23, minute=59, second=59)).order_by('-timestamp')
        # if data_today.count() != 0:
        hours_ago = make_aware(datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(hours=168))
        last_168_data = crypto_model.objects.using(database).filter(timestamp__gte=hours_ago).order_by('timestamp')
        data_sequence_length = crypto_model.objects.using(database).filter(timestamp__lte=item.timestamp).order_by('-timestamp')[:sequence_length]
        close = list(data_sequence_length.values_list('close', flat=True))
        close.append(item.close)
        close = pd.Series(close)
        close_bollinger = close[-bollinger_length:]
        close_rsi = close[-rsi_length:]
        vmap = calculate_vmap(data_day_of_item)
        percentage_returns = calculate_percentage_returns(item, previous_item)
        log_returns = calculate_log_returns(item, previous_item)
        close_higher_shifted_1h = calculate_shifted_value(item, last_168_data, 1)
        close_higher_shifted_24h = calculate_shifted_value(item, last_168_data, 24)
        close_higher_shifted_168h = calculate_shifted_value(item, last_168_data, 168)
        sma = ta.trend.sma_indicator(close, sequence_length)
        ema = ta.trend.ema_indicator(close, sequence_length)
        ema_26 = ta.trend.ema_indicator(close, 26)
        ema_12 = ta.trend.ema_indicator(close, 12)
        macd = ema_12 if ema_12.iloc[-1] != np.nan else 0 - ema_26 if ema_26.iloc[-1]!=np.nan else 0
        bollinger = ta.volatility.BollingerBands(close_bollinger, bollinger_length, 2)
        bollinger_high = bollinger.bollinger_hband()
        bollinger_low = bollinger.bollinger_lband()
        rsi = ta.momentum.rsi(close_rsi, rsi_length)
        item.vmap = vmap
        item.percentage_returns = percentage_returns
        item.log_returns = log_returns
        item.close_higher_shifted_1h = close_higher_shifted_1h
        item.close_higher_shifted_24h = close_higher_shifted_24h
        item.close_higher_shifted_168h = close_higher_shifted_168h
        item.sma = check_nan(sma.iloc[-1])
        item.ema = check_nan(ema.iloc[-1])
        item.macd = check_nan(macd.iloc[-1])
        item.bollinger_high = check_nan(bollinger_high.iloc[-1])
        item.bollinger_low = check_nan(bollinger_low.iloc[-1])
        item.rsi = check_nan(rsi.iloc[-1])
        item.save(using=database)
        previous_item = item
