
from datetime import datetime, timedelta, timezone
from django.utils.timezone import make_aware, make_naive
from django.apps import apps
from .models import Bitcoin, Ethereum, Polkadot, Prediction
from .cb_auth import Granularities
from .views import cb_fetch_product_candles
from stockmarket_bot.celery import app
import numpy as np
import json
from celery import chain
import ta
from django.db.models import Sum, F, FloatField, ExpressionWrapper

def last_full_hour(date_time):
    return date_time.replace(minute=0, second=0, microsecond=0)

@app.task
def update_ohlcv_data():
    cryptos = [Bitcoin, Ethereum, Polkadot]  # Models for cryptos we want to update
    granularity = 3600  # 1 hour in seconds

    for crypto in cryptos:
        latest_entry = crypto.objects.order_by('-timestamp').first()
        # start = latest_entry.timestamp if latest_entry else last_full_hour(datetime.utcnow()) - timedelta(days=30)
        start = make_naive(latest_entry.timestamp) if latest_entry else last_full_hour(datetime.now()) - timedelta(days=30)
        end = last_full_hour(datetime.now())
        delta = end - start
        required_data_points = delta.total_seconds() / granularity
        chunks = -(-required_data_points // 300)  # Calculate the number of chunks (ceiling division)
        # start = int(datetime.timestamp())
        # end = int(datetime.timestamp())
        for _ in range(int(chunks)):
            tmp_end = start + timedelta(hours=300)
            data = cb_fetch_product_candles(f'{crypto.symbol}-USD', int(datetime.timestamp(start)), int(datetime.timestamp(tmp_end)), Granularities.ONE_HOUR.value)
            json_data = json.loads(data.content)
            store_data(crypto, json_data["candles"])
            start += timedelta(hours=300)  # Move start ahead by 300 hours

        # Deleting data older than a month
        # calculate_relational_parameters(crypto)
        month_ago = make_aware(datetime.utcnow() - timedelta(days=30))
        crypto.objects.filter(timestamp__lt=month_ago).delete()
    #! need to update new_data to fetch subset of data and also fetch all crypto data types
    new_data = Bitcoin.objects.all()
    chain(predict_with_lstm.s(new_data), predict_with_xgboost.s(new_data)).apply_async()

def store_data(crypto_model, data):
    for item in data:
        timestamp, low, high, opening, close, volume = item["start"], item["low"], item["high"], item["open"], item["close"], item["volume"]
        sma = ta.trend.sma_indicator(close, 50)
        ema = ta.trend.ema_indicator(close, 50)
        rsi = ta.momentum.rsi(close, 14)
        macd = ta.trend.macd_diff(close)
        bollinger = ta.volatility.BollingerBands(close, 20, 2)
        bollinger_high = bollinger.bollinger_hband()
        bollinger_low = bollinger.bollinger_lband()
        #! set VMAP, percentage returns, log returns and close_higher_shifted_x when all the data is saved!
        data_today = crypto_model.objects.filter(timestamp__gte=timezone.now().replace(hour=0, minute=0, second=0), timestamp__lte=timezone.now().replace(hour=23, minute=59, second=59)).order_by('-timestamp')
        aggregated_data = data_today.aggregate(
            total_volume=Sum('volume'),
            total_value=Sum(
                ExpressionWrapper(
                    ((F('open') + F('close') + F('high')) / 3) * F('volume'),
                    output_field=FloatField()
                )
            )
        )
        if aggregated_data['total_volume'] != 0:
            vmap = aggregated_data['total_value'] / aggregated_data['total_volume'] * volume
        else:
            vmap = 0
        percentage_returns = (close - data_today.last()['close'])/(data_today.last()['close'])
        log_returns = np.log(close - data_today.last()['close'])
        hours_ago = make_aware(datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(hours=168))
        last_168_data = crypto_model.objects.filter(timestamp__gte=hours_ago).order_by('timestamp')
        close_higher_shifted_1h = last_168_data.first()['close'] > close
        close_higher_shifted_168h = last_168_data.last()['close'] > close
        try:
            close_higher_shifted_24h = last_168_data[24]['close'] > close
        except IndexError:
            close_higher_shifted_24h = False
        
        crypto_model.objects.create(
            timestamp=make_aware(datetime.utcfromtimestamp(int(timestamp))),
            open=opening,
            high=high,
            low=low,
            close=close,
            volume=volume,
            sma=sma,
            ema=ema,
            rsi=rsi,
            macd=macd,
            bollinger_low=bollinger_low,
            bollinger_high=bollinger_high,
            vmap=vmap,
            percentage_returns=percentage_returns,
            log_returns=log_returns,
            close_higher_shifted_1h=close_higher_shifted_1h,
            close_higher_shifted_24h=close_higher_shifted_24h,
            close_higher_shifted_168h=close_higher_shifted_168h,
        )

# def calculate_relational_parameters(crypto):
#     hours_ago = make_aware(datetime.utcnow() - timedelta(hours=168))
#     crypto.objects.filter(timestamp__lt=month_ago)
#     last_168_data = crypto.objects.filter()

@app.task
def predict_with_lstm(data):
    # ... logic to predict using LSTM model ...
    lstm_model = apps.get_app_config('coinbase_api').lstm_model
    with no_grad():
        output = lstm_model(data)
        probs = output.tolist()
    print(f'probs: {probs}')
    for idx, item in enumerate(probs):
        pass
        # save the prediction
        # Prediction.objects.create(
        #     timestamp_predicted_for=data['timestamp'],
        #     model_name='LSTM',
        #     predicted_field=f'close_higher_shifted_{1 if idx == 0 else 24 if idx==1 else 168}h',
        #     predicted_value=item
        # )


@app.task
def predict_with_xgboost(data):
    # ... logic to predict using XGBoost model ...
    app_config = apps.get_app_config('coinbase_api')
    xgboost_model1 = app_config.xgboost_model1
    xgboost_model24 = app_config.xgboost_model24
    xgboost_model168 = app_config.xgboost_model168
    # save the prediction
    Prediction.objects.create(
        timestamp_predicted_for=...,
        model_name='XGBoost',
        predicted_field='close',
        predicted_value=...
    )