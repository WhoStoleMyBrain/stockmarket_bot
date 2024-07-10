
from datetime import datetime, timedelta
from django.utils.timezone import make_aware, make_naive
from django.apps import apps

from coinbase_api.enums import Database
from .models.models import AbstractOHLCV, Bitcoin, Ethereum, Polkadot, Prediction
# from .cb_auth import Granularities
from .enums import Granularities
from .utilities.utils import cb_fetch_product_candles
from .utilities.ml_utils import add_calculated_parameters
from stockmarket_bot.celery import app
import json
from torch import no_grad
from coinbase_api.constants import crypto_models


def last_full_hour(date_time):
    return date_time.replace(minute=0, second=0, microsecond=0)

@app.task
def update_ohlcv_data():
    # cryptos = [Bitcoin, Ethereum, Polkadot]  # Models for cryptos we want to update
    cryptos = crypto_models
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
        month_ago = make_aware(datetime.utcnow() - timedelta(days=30))
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
                    timestamp=make_aware(datetime.utcfromtimestamp(int(timestamp))),
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
@app.task
def predict_with_lstm(data, timestamp, crypto_model:AbstractOHLCV, database=Database.DEFAULT.value):
    # print(f'start trying to predict with lstm on database: {database}...')
    lstm_model = apps.get_app_config('coinbase_api').lstm_model
    with no_grad():
        output = lstm_model(data)
        probs = output.tolist()
    # print(f'probs: {probs}')
    for idx, item in enumerate(probs[0]):
        # print(idx, item)
        # save the prediction
        Prediction.objects.using(database).create(
            timestamp_predicted_for=timestamp,
            model_name='LSTM',
            predicted_field=f'close_higher_shifted_{1 if idx == 0 else 24 if idx==1 else 168}h',
            crypto=crypto_model.__name__,
            predicted_value=item
        )


@app.task
def predict_with_xgboost(data,timestamp, crypto_model:AbstractOHLCV, database=Database.DEFAULT.value):
    # ... logic to predict using XGBoost model ...
    # print(f'start trying to predict with xgboost on database: {database}...')
    app_config = apps.get_app_config('coinbase_api')
    xgboost_model1 = app_config.xgboost_model1
    xgboost_model24 = app_config.xgboost_model24
    xgboost_model168 = app_config.xgboost_model168
    y_pred_1 = xgboost_model1.predict(data)
    y_pred_24 = xgboost_model24.predict(data)
    y_pred_168 = xgboost_model168.predict(data)
    # print(f'len data: {data.num_row()}')
    # print(f'data: {data}')
    # print(f'y_pred_1: {len(y_pred_1)}')
    # print(f'y_pred_24: {len(y_pred_24)}')
    # print(f'y_pred_168: {len(y_pred_168)}')
    # save the prediction
    Prediction.objects.using(database).create(
        timestamp_predicted_for=timestamp,
        model_name='XGBoost',
        predicted_field='close_higher_shifted_1h',
        crypto = crypto_model.__name__,
        predicted_value=y_pred_1
    )
    Prediction.objects.using(database).create(
        timestamp_predicted_for=timestamp,
        model_name='XGBoost',
        predicted_field='close_higher_shifted_24h',
        crypto = crypto_model.__name__,
        predicted_value=y_pred_24
    )
    Prediction.objects.using(database).create(
        timestamp_predicted_for=timestamp,
        model_name='XGBoost',
        predicted_field='close_higher_shifted_168h',
        crypto = crypto_model.__name__,
        predicted_value=y_pred_168
    )