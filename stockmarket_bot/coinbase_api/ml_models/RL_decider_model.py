from typing import List
import gymnasium as gym
from gymnasium import spaces
from coinbase_api.utilities.utils import calculate_total_volume
import numpy as np
from coinbase_api.models.models import AbstractOHLCV, Account, Bitcoin, Ethereum, Polkadot, Prediction
from constants import crypto_models, crypto_features, crypto_predicted_features, crypto_extra_features
from coinbase_api.utilities.prediction_handler import PredictionHandler
from ..enums import Database
from enum import Enum
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

    def initial_state(self):
        raise NotImplementedError

class SimulationDataHandler(AbstractDataHandler):
    def __init__(self, initial_volume = 1000):
        self.crypto_models:List[AbstractOHLCV] = crypto_models
        self.total_volume = initial_volume
        self.account_holdings = {crypto.symbol:0 for crypto in self.crypto_models}
        self.crypto_data = self.get_crypto_data()
        self.usdc_held = initial_volume
        self.timestamp = datetime(year=2020, month=1, day=1, hour=0, minute=0, second=0) #! this is the start time
        self.prediction_handler = PredictionHandler(lstm_sequence_length=100, database=Database.SIMULATION.value, timestamp=self.timestamp)

    def prepare_simulation_database(self):
        #! 1. delete all predictions and crypto model entries.
        #! 2. fetch all data up to the current timestamp: crypto models
        #! 3. predict on all data from current timestamp to final entry 
        #!    in historical db at once and save the predictions
        # 1. deletion
        print(f'Preparing simulation.')
        print(f'Deleting predictions')
        predictions = Prediction.objects.using(Database.SIMULATION.value).all()
        predictions.delete()
        for crypto_model in self.crypto_models:
            print(f'Deleting model data for {crypto_model.symbol}')
            crypto_model.objects.using(Database.SIMULATION.value).all().delete()
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
            with transaction.atomic(using=Database.SIMULATION.value):
                crypto_model.objects.using(Database.SIMULATION.value).bulk_create(new_instances)
        # 3. redoing predictions. This is important as models might be retrained!
        #! scratch that, for now predictions will be done live. Might be efficient enough
        # prediction_handler = PredictionHandler(lstm_sequence_length=100, database=Database.SIMULATION.value, timestamp=self.timestamp)
        # total_number_of_data = self.crypto_models[0].objects.using(Database.HISTORICAL.value).filter(timestamp__gte=self.timestamp).count()
        # for i in range(total_number_of_data):
        #     prediction_handler.predict()
        #     prediction_handler.timestamp = prediction_handler.timestamp + timedelta(hours=1)
        # for crypto_model in self.crypto_models:

    def get_new_instance(self, crypto_model:AbstractOHLCV, instance:AbstractOHLCV):
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
    def get_current_state(self):
        # state is:
        # timestamp (internal only, relevant for state changes)
        # total volume
        # account holdings
        # new crypto data
        # eur held (USDC held in the future...)
        
        pass

    def update_state(self, action):
        # update the state based on the action and self.mode
        pass

    def get_crypto_data(self):
        all_entries = []
        for crypto in self.crypto_models:
            crypto_latest = crypto.objects.latest('timestamp')
            all_entries = all_entries + self.crypto_to_list(crypto_latest)
            all_entries = all_entries + self.get_new_prediction_data(crypto, crypto_latest.timestamp)
        return all_entries

    def reset_state(self):
        # return a reset state
        pass

    def initial_state(self):
        # return the initial state
        pass

    def get_crypto_features(self):
        return crypto_features

    def get_crypto_predicted_features(self):
        return crypto_predicted_features
    
    def get_extra_features(self):
        return crypto_extra_features
    
    def crypto_to_list(self, crypto: AbstractOHLCV):
        return [getattr(crypto, fieldname) for fieldname in self.get_crypto_features()]
    
    def get_new_prediction_data(self, crypto_model, timestamp):
        all_entries = []
        ml_models = ['LSTM', 'XGBoost']
        prediction_shifts = [1,24,168]
        for model in ml_models:
            for prediction_shift in prediction_shifts:
                try:
                    entry = Prediction.objects.get(
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

class CustomEnv(gym.Env):
    def __init__(self, data_handler:AbstractDataHandler, asymmetry_factor:float=2):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Assume discrete actions (buy, sell, hold) for simplicity
        # You may need a more complex action space for different assets or continuous actions
        # self.action_space = spaces.Discrete(3)
        self.crypto_models = crypto_models
        self.data_handler = data_handler
        N = len(self.crypto_models)
        self.action_space = spaces.MultiDiscrete([3] * N)  # where N is the number of cryptocurrencies
        self.crypto_data = self.get_crypto_data()
        self.prev_total_volume = None
        self.prev_account_holdings = None
        self.prev_newest_crypto_data = None
        self.prev_eur_held = None
        self.asymmetry_factor = asymmetry_factor
        self.maker_fee = 0.004  # 0.4%
        self.taker_fee = 0.006  # 0.6%
        self.features = self.get_crypto_features()
        self.predicted_features = self.get_crypto_predicted_features()
        self.extra_features = self.get_extra_features()

        M = len(self.features) + len(self.predicted_features) + len(self.extra_features)
        shape_value = M*N + 2 #! +1 because of total volume held and EUR value held
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape_value,), dtype=np.float32)

        # TODO: Load historical data, ML models, etc.

    def step(self, action):
        # save previous values
        # get new values
        total_volume = calculate_total_volume()
        account_holdings = self.get_account_holdings()
        newest_crypto_data = self.get_new_crypto_data()
        eur_held = self.get_eur_held()
        # calculate reward
        reward_q = self.calculate_reward_quadratic(action, total_volume, account_holdings, eur_held)
        # reward_e = self.calculate_reward_exponential(action, total_volume, account_holdings, eur_held)
        # reward_vn = self.calculate_reward_volume_normalized(action, total_volume, account_holdings, eur_held)
        # reward_sharpe = self.calculate_reward_sharpe_ratio(action, total_volume, account_holdings, eur_held)
        self.prev_total_volume = total_volume
        self.prev_account_holdings = account_holdings
        self.prev_eur_held = eur_held
        next_state_concatenated = [total_volume, eur_held] + account_holdings + newest_crypto_data
        print(f'next_state_concatenated: {next_state_concatenated}')
        next_state = next_state_concatenated
        # reward = 0  # Replace with actual reward calculation
        done = False  # Replace with actual termination condition
        info = {}  # Optional: Provide additional information
        return next_state, reward_q, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        # TODO: Reset your system's state
        self.crypto_models = crypto_models
        self.crypto_data = self.get_crypto_data()
        self.prev_total_volume = None
        self.prev_account_holdings = None
        self.prev_newest_crypto_data = None
        self.prev_eur_held = None
        total_volume = calculate_total_volume()
        account_holdings = self.get_account_holdings()
        newest_crypto_data = self.get_new_crypto_data()
        eur_held = self.get_eur_held()
        next_state_concatenated = [total_volume, eur_held] + account_holdings + newest_crypto_data

        self.features =self.get_crypto_features()
        self.predicted_features = self.get_crypto_predicted_features()
        self.extra_features = self.get_extra_features()
        # Mockup for initial_state
        initial_state = next_state_concatenated  # Replace with actual initial state
        return initial_state

    def render(self, mode='human'):
        # Render the environment to the screen or other output
        # Optional: Implement rendering if needed
        pass

    def close(self):
        # Optional: Implement close if any cleanup is needed when the environment is closed
        pass

    def get_crypto_features(self):
        return crypto_features

    def get_crypto_predicted_features(self):
        return crypto_predicted_features
    
    def get_extra_features(self):
        return crypto_extra_features

    def crypto_to_list(self, crypto: AbstractOHLCV):
        return [getattr(crypto, fieldname) for fieldname in self.features]

    def calculate_transaction_volume(self, crypto: AbstractOHLCV, is_buy):
        if is_buy:
            try:
                crypto_account = Account.objects.get(name=f'EUR Wallet')
            except Account.DoesNotExist:
                print('Account with name "EUR Wallet" not found')
                raise Account.DoesNotExist
            transaction_volume = crypto_account.value
        else:
            try:
                crypto_account = Account.objects.get(name=f'{crypto.symbol} Wallet')
            except Account.DoesNotExist:
                print(f'Account with name "{crypto.symbol} Wallet" not found')
                raise Account.DoesNotExist
            price = crypto.objects.latest('timestamp').close
            transaction_volume = price * crypto_account.value
        return transaction_volume  # Assuming this is the result of your method
    
    def cost_for_action(self, action):
        total_cost = 0
        for idx, crypto_action in enumerate(action):
            # crypto, volume, is_buy = crypto_action  # Assuming this structure
            is_buy = True if crypto_action == 2 else False
            print(f'trying to buy {self.crypto_models[idx].__name__}? {is_buy}')
            crypto = self.crypto_models[idx]
            # volume = self.crypto_models[idx]
            try:
                transaction_volume = self.calculate_transaction_volume(crypto, is_buy)
            except Account.DoesNotExist:
                continue
            # Deduct the transaction cost from the reward
            fee_rate = self.maker_fee if is_buy else self.taker_fee  # Assuming maker fee for buy, taker fee for sell
            transaction_cost = transaction_volume * fee_rate
            total_cost += transaction_cost
        return total_cost

    def calculate_reward_quadratic(self, action, total_volume, account_holdings, eur_held):
        if self.prev_total_volume is None:
            # This is the first step, so there's no previous volume to compare to
            return 0

        volume_diff = total_volume - self.prev_total_volume
        if volume_diff > 0:
            reward = volume_diff ** 2
        else:
            reward = -self.asymmetry_factor * (volume_diff ** 2)  # Negative to indicate a punishment
        reward = reward - self.cost_for_action(action)
        return reward
    
    def calculate_reward_exponential(self,action, total_volume, account_holdings, eur_held):
        if self.prev_total_volume is None:
            return 0
        volume_diff = total_volume - self.prev_total_volume
        if volume_diff > 0:
            reward = np.exp(volume_diff) - 1
        else:
            reward = -self.asymmetry_factor * (np.exp(-volume_diff) - 1)
            reward = reward - self.cost_for_action(action)
        return reward
    
    def calculate_reward_volume_normalized(self,action, total_volume, account_holdings, eur_held):
        if self.prev_total_volume is None:
            return 0

        volume_diff = total_volume - self.prev_total_volume
        
        # Assuming you have a method to get the standard deviation of past volume changes
        std_dev = self.exponential_moving_std_dev()  
        
        normalized_diff = volume_diff / (std_dev + 1e-8)  # Adding a small value to avoid division by zero
        
        if normalized_diff > 0:
            reward = normalized_diff ** 2
        else:
            reward = -2 * (normalized_diff ** 2)
            reward = reward - self.cost_for_action(action)

        return reward
    
    def calculate_reward_sharpe_ratio(self,action, total_volume, account_holdings, eur_held):
        # Assuming you have methods to get the expected return and standard deviation of returns
        #TODO finish implementing the submethods for this method
        expected_return = self.get_expected_return()
        std_dev_returns = self.get_std_dev_returns()
        
        # Assuming a constant risk-free rate
        risk_free_rate = 0.01
        
        sharpe_ratio = (expected_return - risk_free_rate) / (std_dev_returns + 1e-8)
        reward = reward - self.cost_for_action(action)
        
        return sharpe_ratio
    
    def exponential_moving_std_dev(self, window):
        # Calculate the Exponential Moving Average (EMA) of the volume
        ema = self.data['volume'].ewm(span=window).mean()
        
        # Calculate the squared differences between the actual values and the EMA
        squared_diff = (self.data['volume'] - ema) ** 2
        
        # Calculate the Exponential Moving Standard Deviation (EMSD)
        emsd = np.sqrt(squared_diff.ewm(span=window).mean())
        
        return emsd.iloc[-1]  # Return the most recent EMSD value
    
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

    def get_account_holdings(self):
        account_holdings = []
        for crypto in self.crypto_models:
            try:
                account = Account.objects.get(name=f'{crypto.symbol} Wallet')
            except Account.DoesNotExist:
                print(f'Account "{crypto.symbol} Wallet" does not exist')
                continue
            account_holdings.append(account.value)
        return account_holdings

    def get_crypto_data(self):
        all_crypto_data = {}
        for crypto in self.crypto_models:
            crypto_data = crypto.objects.all()
            all_crypto_data[crypto.symbol] = crypto_data
        return all_crypto_data
    
    def get_new_crypto_data(self):
        all_entries = []
        for crypto in self.crypto_models:
            crypto_latest = crypto.objects.latest('timestamp')
            all_entries = all_entries + self.crypto_to_list(crypto_latest)
            all_entries = all_entries + self.get_new_prediction_data(crypto, crypto_latest.timestamp)
        return all_entries
    
    def get_new_prediction_data(self, crypto_model, timestamp):
        all_entries = []
        ml_models = ['LSTM', 'XGBoost']
        prediction_shifts = [1,24,168]
        for model in ml_models:
            for prediction_shift in prediction_shifts:
                try:
                    entry = Prediction.objects.get(
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
    
    def get_eur_held(self):
        try:
            eur = Account.objects.get(name='EUR Wallet')
            return eur.value
        except Account.DoesNotExist:
            print('Eur Account does not exist?!?')
            return 0

# Example of how to create an instance of your custom environment
# env = CustomEnv()
