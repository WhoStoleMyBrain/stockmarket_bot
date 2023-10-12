from django.apps import AppConfig
import torch
import xgboost as xgb

from torch import sigmoid
from torch.nn import Module, LSTM, Linear

class StockPredictor(Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(StockPredictor, self).__init__()
        self.lstm = LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.linear = Linear(hidden_dim, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = sigmoid(self.linear(out[:, -1, :]))
        return out

class CoinbaseApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'coinbase_api'
    def ready(self):
        # Load LSTM model
        features = ['Volume USD', 'SMA', 'EMA', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 'VWAP', 'Percentage_Returns', 'Log_Returns']
        TIME_DIFF_1 = 1
        TIME_DIFF_24 = 24 # a day
        TIME_DIFF_168 = 24 * 7 # a week
        targets = [
            f"Target_shifted_{TIME_DIFF_1}",
            f"Target_shifted_{TIME_DIFF_24}",
            f"Target_shifted_{TIME_DIFF_168}"
        ]
        lstm_model = StockPredictor(input_dim=len(features), hidden_dim=50, num_layers=2)  # define the architecture the same way as during training
        # lstm_model = torch.nn.LSTM(...)  # define the architecture the same way as during training
        lstm_model.load_state_dict(torch.load('coinbase_api/ml_models/lstm_model.pth'))
        lstm_model.eval()  # Set the model to evaluation mode

        # Load XGBoost model
        xgboost_model1 = xgb.Booster()
        xgboost_model1.load_model('coinbase_api/ml_models/xgboost_model1.json')

        xgboost_model24 = xgb.Booster()
        xgboost_model24.load_model('coinbase_api/ml_models/xgboost_model24.json')

        xgboost_model168 = xgb.Booster()
        xgboost_model168.load_model('coinbase_api/ml_models/xgboost_model168.json')

        # Set as class attributes
        self.lstm_model = lstm_model
        self.xgboost_model1 = xgboost_model1
        self.xgboost_model24 = xgboost_model24
        self.xgboost_model168 = xgboost_model168
    