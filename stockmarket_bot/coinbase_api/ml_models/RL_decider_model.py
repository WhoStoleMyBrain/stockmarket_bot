from typing import List
import gymnasium as gym
from gymnasium import spaces
from coinbase_api.utilities.utils import calculate_total_volume, initialize_default_cryptos
import numpy as np
from coinbase_api.models.models import AbstractOHLCV, Account, Prediction
from ..constants import crypto_models, crypto_features, crypto_predicted_features, crypto_extra_features
from coinbase_api.utilities.prediction_handler import PredictionHandler
from ..enums import Database, Actions
from datetime import datetime, timedelta
from django.db import transaction

class AbstractDataHandler:
    def __init__(self):
        raise NotImplementedError
        
    def get_current_state(self):
        raise NotImplementedError

    def update_state(self):
        raise NotImplementedError

    def reset_state(self):
        raise NotImplementedError

class SimulationDataHandler(AbstractDataHandler):
    def __init__(self, initial_volume = 1000):
        #! need to also initialize the account table
        self.maker_fee = 0.004  # 0.4%
        self.taker_fee = 0.006  # 0.6%
        self.initial_volume = initial_volume
        self.database = Database.SIMULATION.value
        self.crypto_models:List[AbstractOHLCV] = crypto_models
        self.total_volume = initial_volume
        self.account_holdings = [0 for _ in self.crypto_models]
        self.timestamp = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0) #! this is the start time
        self.prediction_handler = PredictionHandler(lstm_sequence_length=100, database=self.database, timestamp=self.timestamp)
        self.prepare_simulation_database()
    
    def get_current_state(self):
        total_volume = calculate_total_volume(database=self.database)
        account_holdings = self.get_account_holdings()
        new_crypto_data = self.get_new_crypto_data()
        usdc_held = self.get_liquidity()
        return np.array([total_volume, usdc_held] + account_holdings + new_crypto_data)

    def update_state(self, action):
        #! need to perform action, i.e. calculate cost, update account holdings, 
        #!      update crypto database, update predictions (make new ones), update liquidity 
        cost_for_action = self.cost_for_action(action)
        # for efficiency get the indices in the beginning
        buy_indices = [idx for idx, i in enumerate(action) if i == Actions.SELL.value]
        sell_indices = [idx for idx, i in enumerate(action) if i == Actions.BUY.value]
        if len(buy_indices) > 0:
            individual_liquidity = (self.get_liquidity() - cost_for_action)/len(buy_indices)
            for idx in buy_indices:
                # buying... update account holding
                crypto_model = self.crypto_models[idx]
                try:
                    crypto_account = self.get_crypto_account(crypto_model.symbol)
                    usdc_account = self.get_crypto_account('USDC')
                except Account.DoesNotExist:
                    continue
                # get price
                crypto_value = crypto_model.objects.using(self.database).latest('timestamp').close
                crypto_account.value += individual_liquidity/crypto_value
                crypto_account.save(using=self.database)
                usdc_account.value = 0
                usdc_account.save(using=self.database)
        if len(sell_indices) > 0:
            for idx in sell_indices:
                    crypto_model = self.crypto_models[idx]
                    try:
                        crypto_account = self.get_crypto_account(crypto_model.symbol)
                        usdc_account = self.get_crypto_account('USDC')
                    except Account.DoesNotExist:
                        continue
                    # get price
                    crypto_value = crypto_model.objects.using(self.database).latest('timestamp').close
                    total_value = crypto_value * crypto_account.value
                    usdc_account.value += total_value
                    usdc_account.save(using=self.database)
                    crypto_account.value = 0
                    crypto_account.save(using=self.database)
        # fetching new crypto data
        new_timestamp = self.timestamp + timedelta(hours=1)
        # print(f'Stepping from {self.timestamp} to {new_timestamp}')
        done = False
        for crypto in self.crypto_models:
            try:
                historical_data = crypto.objects.using(Database.HISTORICAL.value).get(timestamp=new_timestamp)
            except crypto.DoesNotExist:
                done = True
                print(f'No new data for {crypto.symbol}')
                break
            new_data = self.get_new_instance(crypto_model=crypto, instance=historical_data)
            new_data.save(using=self.database)

        # make new predictions
        self.prediction_handler.timestamp = new_data.timestamp
        self.timestamp = new_timestamp
        self.prediction_handler.predict()
        next_state = self.get_current_state()
        info = {}
        return next_state, cost_for_action, done, info

    def get_crypto_account(self, symbol):
        try:
            crypto_account = Account.objects.using(self.database).get(name=f'{symbol} Wallet')
            return crypto_account
        except Account.DoesNotExist:
            print(f'Account {symbol} Wallet does not exist!')
            raise Account.DoesNotExist

    def reset_state(self):
        self.maker_fee = 0.004  # 0.4%
        self.taker_fee = 0.006  # 0.6%
        self.initial_volume = self.initial_volume
        self.database = Database.SIMULATION.value
        self.crypto_models:List[AbstractOHLCV] = crypto_models
        self.total_volume = self.initial_volume
        self.account_holdings = [0 for _ in self.crypto_models]
        self.timestamp = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0) #! this is the start time
        self.prediction_handler = PredictionHandler(lstm_sequence_length=100, database=self.database, timestamp=self.timestamp)
        self.prepare_simulation_database()
        initial_state = self.get_current_state()
        return initial_state
    
    def cost_for_action(self, action):
        total_cost = 0
        for idx, crypto_action in enumerate(action):
            is_buy = True if crypto_action == Actions.BUY.value else False
            # print(f'trying to buy {self.crypto_models[idx].__name__}? {is_buy}')
            crypto = self.crypto_models[idx]
            try:
                transaction_volume = self.calculate_transaction_volume(crypto, is_buy)
            except Account.DoesNotExist:
                continue
            fee_rate = self.maker_fee if is_buy else self.taker_fee  # Assuming maker fee for buy, taker fee for sell
            transaction_cost = transaction_volume * fee_rate
            total_cost += transaction_cost
        return total_cost
    
    def calculate_transaction_volume(self, crypto: AbstractOHLCV, is_buy):
        if is_buy:
            crypto_account = self.get_crypto_account('USDC')
            transaction_volume = crypto_account.value
        else:
            crypto_account = self.get_crypto_account(crypto.symbol)
            price = crypto.objects.using(self.database).latest('timestamp').close
            transaction_volume = price * crypto_account.value
        return transaction_volume  # Assuming this is the result of your method

    def get_crypto_features(self):
        return crypto_features

    def get_crypto_predicted_features(self):
        return crypto_predicted_features
    
    def get_extra_features(self):
        return crypto_extra_features
    
    def crypto_to_list(self, crypto: AbstractOHLCV):
        return [getattr(crypto, fieldname) for fieldname in self.get_crypto_features()]
    
    def get_new_prediction_data(self, crypto_model:AbstractOHLCV, timestamp:datetime):
        all_entries = []
        ml_models = ['LSTM', 'XGBoost']
        prediction_shifts = [1,24,168]
        for model in ml_models:
            for prediction_shift in prediction_shifts:
                try:
                    entry = Prediction.objects.using(self.database).get(
                        crypto=crypto_model.__name__,
                        timestamp_predicted_for = timestamp, 
                        model_name = model, 
                        predicted_field = f'close_higher_shifted_{prediction_shift}h'
                    )
                    all_entries.append(entry.predicted_value)
                except Prediction.DoesNotExist:
                    print(f'Prediction {crypto_model.__name__}, {timestamp}, {model}, close_higher_shifted_{prediction_shift}h Does not exist')
                    all_entries.append(0)
        return all_entries
    
    def prepare_simulation_database(self):
        #! 1. delete all predictions and crypto model entries.
        #! 2. fetch all data up to the current timestamp: crypto models
        #! 3. prepare the account database
        # 1. deletion
        print(f'Preparing simulation.')
        print(f'Deleting predictions')
        self.prediction_handler.restore_prediction_database()
        for crypto_model in self.crypto_models:
            print(f'Deleting model data for {crypto_model.symbol}')
            crypto_model.objects.using(self.database).all().delete()
        # 2. fetch data
        for crypto_model in self.crypto_models:
            print(f'Fetching model data for {crypto_model.symbol}')
            historical_data = crypto_model.objects.using(Database.HISTORICAL.value).filter(
                timestamp__lte=self.timestamp,
                timestamp__gte=self.timestamp - timedelta(days=30)
            )
            new_instances = []
            for obj in historical_data:
                #TODO Improve model saving...
                new_instance = self.get_new_instance(crypto_model, obj)
                new_instances.append(new_instance)
            with transaction.atomic(using=self.database):
                crypto_model.objects.using(self.database).bulk_create(new_instances)
        # 3. prepare the account database
        self.reset_account_data()
        # 4. get initial predictions in
        self.prediction_handler.predict()

    def reset_account_data(self):
        account_data = Account.objects.using(self.database).all()
        account_data.delete()
        initialize_default_cryptos(initial_volume=self.initial_volume, database=self.database)
        

    def get_new_instance(self, crypto_model:AbstractOHLCV, instance:AbstractOHLCV) -> AbstractOHLCV:
        return crypto_model(
                    timestamp=instance.timestamp,
                    open=instance.open,
                    high=instance.high,
                    low=instance.low,
                    close=instance.close,
                    volume=instance.volume,
                    sma=instance.sma,
                    ema=instance.ema,
                    rsi=instance.rsi,
                    macd=instance.macd,
                    bollinger_high=instance.bollinger_high,
                    bollinger_low=instance.bollinger_low,
                    vmap=instance.vmap,
                    percentage_returns=instance.percentage_returns,
                    log_returns=instance.log_returns,
                    close_higher_shifted_1h=instance.close_higher_shifted_1h,
                    close_higher_shifted_24h=instance.close_higher_shifted_24h,
                    close_higher_shifted_168h=instance.close_higher_shifted_168h,
                )
    
    def get_liquidity(self):
        try:
            usdc_account = self.get_crypto_account('USDC')
            return usdc_account.value
        except Account.DoesNotExist:
            print('USDC Account does not exist?!?')
            return 0
        
    def get_account_holdings(self):
        account_holdings = []
        for crypto in self.crypto_models:
            try:
                account = Account.objects.using(self.database).get(name=f'{crypto.symbol} Wallet')
            except Account.DoesNotExist:
                print(f'Account "{crypto.symbol} Wallet" does not exist')
                continue
            account_holdings.append(account.value)
        return account_holdings
    
    def get_new_crypto_data(self):
        all_entries = []
        for crypto in self.crypto_models:
            crypto_latest = crypto.objects.using(self.database).latest('timestamp')
            all_entries = all_entries + self.crypto_to_list(crypto_latest)
            all_entries = all_entries + self.get_new_prediction_data(crypto, crypto_latest.timestamp)
        return all_entries

class CustomEnv(gym.Env):
    def __init__(self, data_handler:AbstractDataHandler, asymmetry_factor:float=2):
        print('Initializing env')
        super(CustomEnv, self).__init__()
        self.crypto_models = crypto_models
        self.data_handler = data_handler
        N = len(self.crypto_models)
        self.action_space = spaces.MultiDiscrete([3] * N)  # where N is the number of cryptocurrencies
        self.prev_total_volume = None
        self.asymmetry_factor = asymmetry_factor
        self.volume_timeframe= 24*3
        self.volume_values = [0]*self.volume_timeframe
        # self.maker_fee = 0.004  # 0.4%
        # self.taker_fee = 0.006  # 0.6%
        self.features = self.get_crypto_features()
        self.predicted_features = self.get_crypto_predicted_features()
        self.extra_features = self.get_extra_features()
        M = len(self.features) + len(self.predicted_features) + len(self.extra_features)
        shape_value = M*N + 2 #! +1 because of total volume held and USDC value held
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape_value,), dtype=np.float64)

    def step(self, action):
        # print(f'stepping env with action: {action}')
        next_state, cost_for_action, terminated, info = self.data_handler.update_state(action)
        total_volume = next_state[0]
        self.volume_values = self.volume_values[1:] + [total_volume]
        usdc_held = next_state[1]
        reward_q = self.calculate_reward_quadratic(action, total_volume, cost_for_action)
        self.prev_total_volume = total_volume
        # print(f'next_state: {next_state}')
        truncated = False
        print(f'current volume: {total_volume}, usdc: {usdc_held}, reward {reward_q}, cost: {cost_for_action}')
        
        return next_state, reward_q, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # print(f'received additional params on reset: seed={seed}, options={options}')
        self.crypto_models = crypto_models
        self.prev_total_volume = None
        self.volume_timeframe= 24*3
        self.volume_values = [0]*self.volume_timeframe
        self.features =self.get_crypto_features()
        self.predicted_features = self.get_crypto_predicted_features()
        self.extra_features = self.get_extra_features()
        initial_state = self.data_handler.reset_state()
        info = {}
        return initial_state, info

    def render(self, mode='human'):
        print('trying to render...')
        # Render the environment to the screen or other output
        # Optional: Implement rendering if needed
        pass

    def close(self):
        print('trying to close...')
        # Optional: Implement close if any cleanup is needed when the environment is closed
        pass

    def get_crypto_features(self):
        return crypto_features

    def get_crypto_predicted_features(self):
        return crypto_predicted_features
    
    def get_extra_features(self):
        return crypto_extra_features

    def calculate_reward_quadratic(self, action, total_volume, cost_for_action):
        #todo redo quadratic, made linear for now
        if self.prev_total_volume is None:
            # This is the first step, so there's no previous volume to compare to
            return 0

        volume_diff = total_volume - self.prev_total_volume
        if volume_diff > 0:
            reward = volume_diff
        else:
            reward = -self.asymmetry_factor * (volume_diff)  # Negative to indicate a punishment
        reward = reward - cost_for_action
        return reward
    
    def calculate_reward_exponential(self, action, total_volume, cost_for_action):
        if self.prev_total_volume is None:
            return 0
        volume_diff = total_volume - self.prev_total_volume
        if volume_diff > 0:
            reward = np.exp(volume_diff) - 1
        else:
            reward = -self.asymmetry_factor * (np.exp(-volume_diff) - 1)
            reward = reward - cost_for_action
        return reward
    
    def calculate_reward_volume_normalized(self, action, total_volume, cost_for_action):
        if self.prev_total_volume is None:
            return 0
        volume_diff = total_volume - self.prev_total_volume
        # Assuming you have a method to get the standard deviation of past volume changes
        # need to fetch values for exp std_dev
        # values = 
        std_dev = self.exponential_moving_std_dev()  
        normalized_diff = volume_diff / (std_dev + 1e-8)  # Adding a small value to avoid division by zero
        if normalized_diff > 0:
            reward = normalized_diff ** 2
        else:
            reward = -self.asymmetry_factor * (normalized_diff ** 2)
            reward = reward - cost_for_action
        return reward
    
    def calculate_reward_sharpe_ratio(self,action, total_volume, cost_for_action):
        # Assuming you have methods to get the expected return and standard deviation of returns
        #TODO finish implementing the submethods for this method
        expected_return = self.get_expected_return()
        std_dev_returns = self.get_std_dev_returns()
        
        # Assuming a constant risk-free rate
        risk_free_rate = 0.01
        
        sharpe_ratio = (expected_return - risk_free_rate) / (std_dev_returns + 1e-8)
        reward = reward - cost_for_action
        
        return sharpe_ratio
    
    def calculate_ema(self, values, window):
        alpha = 2 / (window + 1)
        ema_values = [values[0]]  # Initialize with the first value
        for val in values[1:]:
            next_ema = (1 - alpha) * ema_values[-1] + alpha * val
            ema_values.append(next_ema)
        return ema_values
    
    def exponential_moving_std_dev(self):
        ema = self.calculate_ema(self.volume_values, self.volume_timeframe)
        # squared_diff = (self.data['volume'] - ema) ** 2
        
        # emsd = np.sqrt(squared_diff.ewm(span=self.volume_timeframe).mean())
        
        # return emsd.iloc[-1]  # Return the most recent EMSD value
        squared_diff = [(val - ema[i]) ** 2 for i, val in enumerate(self.volume_values)]
        
        emsd_values = self.calculate_ema(squared_diff, self.volume_timeframe)
        
        return emsd_values[-1] ** 0.5  # Square root of the most recent EMSD value
    
    def get_expected_return(self):
        #TODO implement this method
        # This is a simplified example. Replace with your method of calculating expected return.
        # Assume daily returns are stored in a column called 'daily_return'
        # expected_return = self.data['daily_return'].mean()
        
        return 0
    
    def get_std_dev_returns(self):
        #TODO implement this method
        # This is a simplified example. Replace with your method of calculating standard deviation of returns.
        # std_dev_returns = self.data['daily_return'].std()
        
        return 0

    