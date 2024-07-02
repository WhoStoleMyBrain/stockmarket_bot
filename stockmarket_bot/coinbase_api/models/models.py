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
    sma = models.FloatField(null=True, default=0)
    ema = models.FloatField(null=True, default=0)
    rsi = models.FloatField(null=True, default=0)
    macd = models.FloatField(null=True, default=0)
    bollinger_high = models.FloatField(null=True, default=0)
    bollinger_low = models.FloatField(null=True, default=0)
    vmap = models.FloatField(null=True, default=0)
    percentage_returns = models.FloatField(null=True, default=0)
    log_returns = models.FloatField(null=True, default=0)
    close_higher_shifted_1h = models.BooleanField(null=True)
    close_higher_shifted_24h = models.BooleanField(null=True)
    close_higher_shifted_168h = models.BooleanField(null=True)

    # timestamp, open, high, low, 

    

    class Meta:
        abstract = True
        indexes = [
            # models.Index(fields=['timestamp', 'high']),  # Composite index example
            # models.Index(fields=['timestamp', 'low']),   # Composite index example
            models.Index(fields=['timestamp', 'close']),   # Composite index example
            # ... add more indexes as needed ...
        ]

    def all_fields_set(self)->bool:
        # TODO redo this method since there is actually no none values anymore
        # print(f'check in all_fields_set. self: {self}')
        if (
            self.open is None or
            self.high is None or
            self.low is None or
            self.close is None or
            self.volume is None or
            self.sma is None or
            self.ema is None or
            self.rsi is None or
            self.macd is None or
            self.bollinger_high is None or
            self.bollinger_low is None or
            self.percentage_returns is None or
            self.log_returns is None or
            self.close_higher_shifted_1h is None or
            self.close_higher_shifted_24h is None or
            self.close_higher_shifted_168h is None or
            self.open == 0.0 or
            self.high == 0.0 or
            self.low == 0.0 or
            self.close == 0.0 or
            self.volume == 0.0 or
            self.sma == 0.0 or
            self.ema == 0.0 or
            self.rsi == 0.0 or
            self.macd == 0.0 or
            self.bollinger_high == 0.0 or
            self.bollinger_low == 0.0 or
            self.percentage_returns == 0.0 or
            self.log_returns == 0.0         
        ):
            return False
        return True

    @staticmethod
    def queryset_to_lstm_dataframe(queryset, seq_length=100):
        features = ['volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns']
        # Convert the queryset to a list of dictionaries
        data_dict_list = queryset.values()
        # print(f'data dict list: {len(data_dict_list)}')
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
        # features_extended = ['volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns', 'Hour', 'Day_of_Week', 'Day_of_Month', 'Month', 'Year', 'Is_Weekend']
        drop_features = ['macd', 'sma', 'Day_of_Week', 'Day_of_Month', 'Hour', 'log_returns', 'Is_Weekend']
        # drop_features = ['macd', 'sma', 'Day_of_Week', 'Day_of_Month', 'Hour', 'log_returns', 'Is_Weekend']
        features_extended = [feature for feature in features_extended if feature not in drop_features]
        # features_extended = ['volume', 'ema', 'rsi', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'Month', 'Year']
        # dataframe columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap','percentage_returns', 'log_returns', 'close_higher_shifted_1h','close_higher_shifted_24h', 'close_higher_shifted_168h', 'Datetime','Hour', 'Day_of_Week', 'Day_of_Month', 'Month', 'Year', 'Is_Weekend']
        # print(f'features extended: {features_extended}')
        # print(dataframe.head())
        # print(dataframe.columns)
        X_extended = dataframe[features_extended].values
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_extended)
        data_normalized = dataframe.copy()
        data_normalized[features_extended] = X_normalized
        prices = data_normalized[features_extended].values
        tmp = xgb.DMatrix(prices)
        return tmp
    
    @classmethod
    def default_entry(cls, timestamp):
        return cls(
            timestamp = timestamp,
            open = 0,
            high = 0,
            low = 0,
            close = 0,
            volume = 0,
            sma = 0,
            ema = 0,
            rsi = 0,
            macd = 0,
            bollinger_high = 0,
            bollinger_low = 0,
            vmap = 0,
            percentage_returns = 0,
            log_returns = 0,
            close_higher_shifted_1h = 0,
            close_higher_shifted_24h = 0,
            close_higher_shifted_168h = 0
        )

    def set_nonset_values_to_default(self):
        self.open = 0.0 if (self.open == None or self.open == 0.0) else self.open
        self.high = 0.0 if (self.high == None or self.high == 0.0) else self.high
        self.low = 0.0 if (self.low == None or self.low == 0.0) else self.low
        self.close = 0.0 if (self.close == None or self.close == 0.0) else self.close
        self.volume = 0.0 if (self.volume == None or self.volume == 0.0) else self.volume
        self.sma = 0.0 if (self.sma == None or self.sma == 0.0) else self.sma
        self.ema = 0.0 if (self.ema == None or self.ema == 0.0) else self.ema
        self.rsi = 0.0 if (self.rsi == None or self.rsi == 0.0) else self.rsi
        self.macd = 0.0 if (self.macd == None or self.macd == 0.0) else self.macd
        self.bollinger_high = 0.0 if (self.bollinger_high == None or self.bollinger_high == 0.0) else self.bollinger_high
        self.bollinger_low = 0.0 if (self.bollinger_low == None or self.bollinger_low == 0.0) else self.bollinger_low
        self.vmap = 0.0 if (self.vmap == None or self.vmap == 0.0) else self.vmap
        self.percentage_returns = 0.0 if (self.percentage_returns == None or self.percentage_returns == 0.0) else self.percentage_returns
        self.log_returns = 0.0 if (self.log_returns == None or self.log_returns == 0.0) else self.log_returns
        self.close_higher_shifted_1h = 0.0 if (self.close_higher_shifted_1h == None or self.close_higher_shifted_1h == 0.0) else self.close_higher_shifted_1h
        self.close_higher_shifted_24h = 0.0 if (self.close_higher_shifted_24h == None or self.close_higher_shifted_24h == 0.0) else self.close_higher_shifted_24h
        self.close_higher_shifted_168h = 0.0 if (self.close_higher_shifted_168h == None or self.close_higher_shifted_168h == 0.0) else self.close_higher_shifted_168h
        return self
        

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
    
class Solana(AbstractOHLCV):
    symbol = "SOL"
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
    symbol = models.CharField(max_length=255, unique=True)
    earliest_date = models.DateTimeField()

    def __str__(self):
        return self.symbol

    @staticmethod
    def symbol_to_storage(symbol):
        return f'{symbol}-USDC'
    
    @staticmethod
    def stored_symbol_to_model(model_entry):
        return model_entry.symbol.split('-')[0]