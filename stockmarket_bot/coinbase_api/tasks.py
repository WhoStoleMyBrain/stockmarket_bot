
from datetime import datetime, timedelta
from django.utils.timezone import make_aware, make_naive
from django.apps import apps
from .models import Bitcoin, Ethereum, Polkadot, Prediction
from .cb_auth import Granularities
from .utilities.utils import cb_fetch_product_candles
from .utilities.ml_utils import add_calculated_parameters
from stockmarket_bot.celery import app
import json
from torch import no_grad


def last_full_hour(date_time):
    return date_time.replace(minute=0, second=0, microsecond=0)

@app.task
def update_ohlcv_data():
    cryptos = [Bitcoin, Ethereum, Polkadot]  # Models for cryptos we want to update
    granularity = 3600  # 1 hour in seconds

    for crypto in cryptos:
        print(f'Starting with crypto: {crypto.__name__}')
        #! delete the following two lines, only keep if need to redo database arises
        all_data = crypto.objects.all()
        all_data.delete()

        latest_entry = crypto.objects.order_by('-timestamp').first()
        # start = latest_entry.timestamp if latest_entry else last_full_hour(datetime.utcnow()) - timedelta(days=30)
        start = make_naive(latest_entry.timestamp) if latest_entry else last_full_hour(datetime.now()) - timedelta(days=30)
        end = last_full_hour(datetime.now())
        delta = end - start
        required_data_points = delta.total_seconds() / granularity
        chunks = -(-required_data_points // 300)  # Calculate the number of chunks (ceiling division)
        for _ in range(int(chunks)):
            print(f'starting with chunk {_+1} of {int(chunks)}')
            tmp_end = start + timedelta(hours=300)
            data = cb_fetch_product_candles(f'{crypto.symbol}-USD', int(datetime.timestamp(start)), int(datetime.timestamp(tmp_end)), Granularities.ONE_HOUR.value)
            json_data = json.loads(data.content)
            store_data(crypto, json_data["candles"])
            start += timedelta(hours=300)  # Move start ahead by 300 hours
            print(f'finished with chunk {_+1} of {int(chunks)}')

        # Deleting data older than a month
        add_calculated_parameters(crypto)
        month_ago = make_aware(datetime.utcnow() - timedelta(days=30))
        crypto.objects.filter(timestamp__lt=month_ago).delete()
        print(f'Finished with crypto: {crypto.__name__}, {crypto._meta.model_name}')
    #! need to update new_data to fetch subset of data and also fetch all crypto data types
    # new_data = Bitcoin.objects.all()
    # chain(predict_with_lstm.s(new_data), predict_with_xgboost.s(new_data)).apply_async()

def store_data(crypto_model, data):
    try:
        for item in data:
            timestamp, low, high, opening, close_base, volume = float(item["start"]), float(item["low"]), float(item["high"]), float(item["open"]), float(item["close"]), float(item["volume"])
            new_entry = crypto_model.objects.create(
                timestamp=make_aware(datetime.utcfromtimestamp(int(timestamp))),
                open=opening,
                high=high,
                low=low,
                close=close_base,
                volume=volume,
            )
            new_entry.save()
    except Exception as e:
        print(f'Encountered the following error: {e}')

@app.task
def predict_with_lstm(data):
    # ... logic to predict using LSTM model ...
    print('start trying to predict with lstm...')
    lstm_model = apps.get_app_config('coinbase_api').lstm_model
    # print(f'lstm_model: {lstm_model}')
    # for name, param in lstm_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    # print(f'data: {data}')
    with no_grad():
        output = lstm_model(data)
        probs = output.tolist()
    print(f'probs: {probs}')
    for idx, item in enumerate(probs):
        print(idx, item)
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
    print('start trying to predict with xgboost...')
    app_config = apps.get_app_config('coinbase_api')
    xgboost_model1 = app_config.xgboost_model1
    xgboost_model24 = app_config.xgboost_model24
    xgboost_model168 = app_config.xgboost_model168
    y_pred_1 = xgboost_model1.predict(data)
    y_pred_24 = xgboost_model24.predict(data)
    y_pred_168 = xgboost_model168.predict(data)
    print(f'len data: {data.num_row()}')
    print(f'y_pred_1: {len(y_pred_1)}')
    print(f'y_pred_24: {len(y_pred_24)}')
    print(f'y_pred_168: {len(y_pred_168)}')
    # save the prediction
    # Prediction.objects.create(
    #     timestamp_predicted_for=...,
    #     model_name='XGBoost',
    #     predicted_field='close',
    #     predicted_value=...
    # )


# SELECT * FROM coinbase_api_bitcoin
# WHERE 
#     open = 'nan' OR
#     high = 'nan' OR
#     low = 'nan' OR
#     close = 'nan' OR
#     volume = 'nan' OR
#     sma = 'nan' OR
#     ema = 'nan' OR
#     rsi = 'nan' OR
#     macd = 'nan' OR
#     bollinger_high = 'nan' OR
#     bollinger_low = 'nan' OR
#     vmap = 'nan' OR
#     percentage_returns = 'nan' OR
#     log_returns = 'nan';
