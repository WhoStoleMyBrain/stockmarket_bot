from django.db import models
import pandas as pd
from torch import no_grad, tensor, float32
import numpy as np
import xgboost as xgb

class Cryptocurrency(models.Model):
    base_display_symbol = models.CharField(max_length=255)  # For storing the symbol of the base (e.g. BTC)
    quote_display_symbol = models.CharField(max_length=255)  # For storing the symbol of the quote (e.g. USD / EUR)
    product_id = models.CharField(max_length=64, unique=True)  # For storing the symbol of the cryptocurrency, e.g., BTC.
    trading_indicator = models.FloatField(default=0)  # A number between 0 and 1 as you mentioned.
    # Add other fields that you retrieve from the API if necessary
    def __str__(self):
        return self.product_id
    class Meta:
        unique_together = [
            ["base_display_symbol", "quote_display_symbol"]
        ]

class AbstractOHLCV(models.Model):
    timestamp = models.DateTimeField(db_index=True)
    open = models.FloatField(null=True)
    high = models.FloatField(null=True, db_index=True)  # Assuming high/low might be queried often
    low = models.FloatField(null=True, db_index=True)   # Assuming high/low might be queried often
    close = models.FloatField(null=True, db_index=True)
    volume = models.FloatField(null=True)
    sma = models.FloatField(null=True)
    ema = models.FloatField(null=True)
    rsi = models.FloatField(null=True)
    macd = models.FloatField(null=True)
    bollinger_high = models.FloatField(null=True)
    bollinger_low = models.FloatField(null=True)
    vmap = models.FloatField(null=True)
    percentage_returns = models.FloatField(null=True)
    log_returns = models.FloatField(null=True)
    close_higher_shifted_1h = models.BooleanField(null=True)
    close_higher_shifted_24h = models.BooleanField(null=True)
    close_higher_shifted_168h = models.BooleanField(null=True)

    class Meta:
        abstract = True
        indexes = [
            # models.Index(fields=['timestamp', 'high']),  # Composite index example
            # models.Index(fields=['timestamp', 'low']),   # Composite index example
            models.Index(fields=['timestamp', 'close']),   # Composite index example
            # ... add more indexes as needed ...
        ]

    @staticmethod
    def queryset_to_lstm_dataframe(queryset, seq_length=100):
        features = ['volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns']
        # Convert the queryset to a list of dictionaries
        data_dict_list = queryset.values()
        # print(data_dict_list)
        dataframe = pd.DataFrame.from_records(data_dict_list, index='timestamp')
        dataframe.drop(columns=['id'], inplace=True)
        prices = dataframe[features].values
        # print(prices[-5:])
        X = []
        # for i in range(len(prices) - seq_length):
        X.append(prices[:, :len(features)])
            # y_values = [prices[i+seq_length, len(features)+target_idx] for target_idx in range(len(targets))]
            # y.append(y_values)
        # Convert the list of dictionaries to a DataFrame
        tensor_data = tensor(X, dtype=float32)
        return tensor_data
    
    @staticmethod
    def queryset_to_xgboost_dataframe(queryset):
        def check_timestamp(ts):
            return len(str(ts)) == 13
        features = ['volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns']
        # Convert the queryset to a list of dictionaries
        data_dict_list = queryset.values()
        # print(data_dict_list)
        dataframe = pd.DataFrame.from_records(data_dict_list)
        dataframe.drop(columns=['id'], inplace=True)
        dataframe['timestamp'] = dataframe['timestamp'].apply(lambda x: x//1000 if check_timestamp(x) else x)
        dataframe['Datetime'] = pd.to_datetime(dataframe['timestamp'], unit='s')
        dataframe['Hour'] = dataframe['Datetime'].dt.hour
        dataframe['Day_of_Week'] = dataframe['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
        dataframe['Day_of_Month'] = dataframe['Datetime'].dt.day
        dataframe['Month'] = dataframe['Datetime'].dt.month
        dataframe['Year'] = dataframe['Datetime'].dt.year
        dataframe['Is_Weekend'] = (dataframe['Day_of_Week'] >= 5).astype(int)  # 1 for weekend, 0 for weekdays
        # Updating the features list
        features_extended = features + ['Hour', 'Day_of_Week', 'Day_of_Month', 'Month', 'Year', 'Is_Weekend']
        #  ('MACD', 0.045680176),
        #  ('SMA', 0.039496846),
        #  ('Day_of_Week', 0.038991235),
        #  ('Day_of_Month', 0.038741197),
        #  ('Hour', 0.03847502),
        #  ('Log_Returns', 0.0),
        #  ('Is_Weekend', 0.0)]
        drop_features = ['macd', 'sma', 'Day_of_Week', 'Day_of_Month', 'Hour', 'log_returns', 'Is_Weekend']
        features_extended = [feature for feature in features_extended if feature not in drop_features]
        print(f'len features: {len(features)}; len features extended: {len(features_extended)}')
        prices = dataframe[features_extended].values
        # X = prices[:, :-3]  # Features
        # X = np.array(prices).reshape((1, -1))
        X = prices
        tmp = xgb.DMatrix(X)
        # print(prices.tail())
        return tmp
        

class Bitcoin(AbstractOHLCV):
    symbol = "BTC"
    def __str__(self) -> str:
        return self.symbol

class Ethereum(AbstractOHLCV):
    symbol = "ETH"
    def __str__(self) -> str:
        return self.symbol

class Polkadot(AbstractOHLCV):
    symbol = "DOT"
    def __str__(self) -> str:
        return self.symbol
    
class Prediction(models.Model):
    timestamp_predicted_for = models.DateTimeField(db_index=True)
    timestamp_predicted_at = models.DateTimeField(auto_now_add=True, db_index=True)
    model_name = models.CharField(max_length=255, db_index=True)  # e.g. 'LSTM', 'XGBoost'
    predicted_field = models.CharField(max_length=50, db_index=True)  # e.g. 'open', 'close', 'high', 'low', 'volume'
    predicted_value = models.FloatField(null=True)
    # any other fields you need...

    class Meta:
        unique_together = ['timestamp_predicted_for', 'model_name', 'predicted_field']  # To ensure unique combination of these fields
