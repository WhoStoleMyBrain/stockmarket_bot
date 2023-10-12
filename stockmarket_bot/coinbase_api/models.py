from django.db import models

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
    bollinger_low = models.FloatField(null=True)
    bollinger_high = models.FloatField(null=True)
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
