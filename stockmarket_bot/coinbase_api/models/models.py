from django.db import models
import pandas as pd
from torch import no_grad, tensor, float32
from sklearn.preprocessing import StandardScaler
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
    symbol = "AAA"
    def __str__(self) -> str:
        return self.symbol
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
        dataframe = pd.DataFrame.from_records(data_dict_list, index='timestamp')
        dataframe.drop(columns=['id'], inplace=True)
        dataframe = dataframe.fillna(0)
        # print(dataframe.head())
        # print(dataframe.tail())
        prices = dataframe[features].values
        scaler = StandardScaler()
        scaler.fit(prices)
        X_test = []
        X_test.append(prices[:, :len(features)])
        X_test = np.array(X_test)
        X_test_2D = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled_2D = scaler.transform(X_test_2D)
        X_test = X_test_scaled_2D.reshape(X_test.shape)
        tensor_data = tensor(X_test, dtype=float32)
        return tensor_data
    
    @staticmethod
    def queryset_to_xgboost_dataframe(queryset):
        def check_timestamp(ts):
            return len(str(ts)) == 13
        features = ['volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns']
        # Convert the queryset to a list of dictionaries
        data_dict_list = queryset.values()
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
        drop_features = ['macd', 'sma', 'Day_of_Week', 'Day_of_Month', 'Hour', 'log_returns', 'Is_Weekend']
        features_extended = [feature for feature in features_extended if feature not in drop_features]
        X_extended = dataframe[features_extended].values
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_extended)
        data_normalized = dataframe.copy()
        data_normalized[features_extended] = X_normalized
        prices = data_normalized[features_extended].values
        tmp = xgb.DMatrix(prices)
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
    crypto = models.CharField(max_length=50, db_index=True, default='Bitcoin')
    predicted_value = models.FloatField(null=True)
    # any other fields you need...

    class Meta:
        unique_together = ['timestamp_predicted_for', 'model_name', 'predicted_field', 'crypto']  # To ensure unique combination of these fields

class Account(models.Model):
    name = models.CharField(max_length=255)
    uuid = models.UUIDField()
    currency = models.CharField(max_length=10)
    value = models.FloatField()

class CryptoMetadata(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    earliest_date = models.DateTimeField()

    def __str__(self):
        return self.symbol

    @staticmethod
    def symbol_to_storage(symbol):
        return f'{symbol}-USDC'
    
    @staticmethod
    def stored_symbol_to_model(model_entry):
        return model_entry.symbol.split('-')[0]