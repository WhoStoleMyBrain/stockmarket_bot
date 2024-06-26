from abc import abstractmethod
from typing import Any, Dict, List, Tuple
import gymnasium as gym
from gymnasium import spaces
from coinbase_api.utilities.utils import calculate_total_volume, initialize_default_cryptos
import numpy as np
import numpy.typing as npt
from coinbase_api.models.models import AbstractOHLCV, Account, Prediction, CryptoMetadata
from ..constants import crypto_models, crypto_features, crypto_predicted_features, crypto_extra_features
from coinbase_api.utilities.prediction_handler import PredictionHandler
from ..enums import Database, Actions
from datetime import datetime, timedelta
from django.db import transaction
import random

class AbstractDataHandler:
    def __init__(self):
        raise NotImplementedError
        
    def get_current_state(self) -> npt.NDArray[np.float16]:
        raise NotImplementedError

    def update_state(self, action: Actions) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        raise NotImplementedError

    def reset_state(self) -> npt.NDArray[np.float16]:
        raise NotImplementedError

    def get_reward_ratios_for_current_timestep(self) -> Dict[str, float]:
        raise NotImplementedError

class SimulationDataHandler(AbstractDataHandler):
    def __init__(self, initial_volume = 1000) -> None:
        self.initial_timestamp = None
        
        #! need to also initialize the account table
        self.maker_fee = 0.004  # 0.4%
        self.taker_fee = 0.006  # 0.6%
        self.initial_volume: float = initial_volume
        self.database: Database = Database.SIMULATION.value
        self.crypto_models:List[AbstractOHLCV] = crypto_models
        self.total_volume = initial_volume
        self.account_holdings = [0 for _ in self.crypto_models]
        self.earliest_timestamp = self.get_earliest_timestamp()
        self.maximum_timestamp = self.get_maximum_timestamp()
        self.buffer = 1024
        self.initial_timestamp = self.get_starting_timestamp()
        self.timestamp = self.initial_timestamp
        self.initial_prices = self.get_initial_crypto_prices()
        # self.timestamp = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0) #! this is the start time
        self.prediction_handler = PredictionHandler(lstm_sequence_length=100, database=self.database, timestamp=self.timestamp)
        self.prepare_simulation_database()


    def get_initial_crypto_prices(self) -> Dict[str, float]:
        values = {}
        for crypto_model in self.crypto_models:
            values[crypto_model.symbol] = crypto_model.objects.using(Database.HISTORICAL.value).filter(timestamp=self.timestamp).first().close
        # print(f'found initial crypto prices: {values}. timestamp: {self.timestamp}')
        return values

    def get_reward_ratios_for_current_timestep(self) -> Dict[str, float]:
        ratios = {}
        for crypto_model in self.crypto_models:
            value = crypto_model.objects.using(Database.HISTORICAL.value).filter(timestamp=self.timestamp).first().close
            ratios[crypto_model.symbol] = value / self.initial_prices[crypto_model.symbol]
            # print(f'ratio = {value}/{self.initial_prices[crypto_model.symbol]} = {ratios[crypto_model.symbol]}')
        # print(f'reward ratios for stepcount {self.step_count}: {ratios}. timestamp: {self.timestamp}')
        return ratios

    def get_earliest_timestamp(self) -> datetime:
        all_information = CryptoMetadata.objects.using(Database.HISTORICAL.value).all()
        timestamps = all_information.values_list("earliest_date").order_by("-earliest_date")
        return timestamps.first()[0]
    
    def get_maximum_timestamp(self) -> datetime:
        all_timestamps = []
        for crypto_model in self.crypto_models:
            all_timestamps.append(crypto_model.objects.using(Database.HISTORICAL.value).values_list("timestamp").order_by("-timestamp").first())
        all_timestamps.sort()
        return all_timestamps[0][0]
    
    def get_starting_timestamp(self) -> datetime:
        if (self.initial_timestamp) is None:
            hours = self.maximum_timestamp - self.earliest_timestamp
            hours_number = hours.total_seconds() // 3600 # hours between min and max ts
            rand_start = random.randint(0, hours_number - self.buffer)
            print(f'starting timestamp: {self.earliest_timestamp + timedelta(hours=rand_start)}\tminimum: {self.earliest_timestamp}\tmaximum: {self.maximum_timestamp}\t random number: {rand_start}/{int(hours_number)}')
            return self.earliest_timestamp + timedelta(hours=rand_start)
        print('initial timestamp was already set')
        return self.initial_timestamp
    
    def get_current_state(self) -> npt.NDArray[np.float16]:
        total_volume = calculate_total_volume(database=self.database)
        account_holdings = self.get_account_holdings()
        new_crypto_data = self.get_new_crypto_data()
        usdc_held = self.get_liquidity()
        # print(f'lengths in current state: account holdings: {len(account_holdings)}; new_crypto_data: {len(new_crypto_data)}')
        return np.array([total_volume, usdc_held] + account_holdings + new_crypto_data)

    def update_state(self, action: Actions) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        # next_state, cost_for_action, done, info
        cost_for_action = self.cost_for_action(action)
        buy_indices = [idx for idx, i in enumerate(action) if i == Actions.SELL.value]
        sell_indices = [idx for idx, i in enumerate(action) if i == Actions.BUY.value]
        # perform buy actions on DB
        if len(buy_indices) > 0:
            individual_liquidity = (self.get_liquidity() - cost_for_action)/len(buy_indices)
            for idx in buy_indices:
                crypto_model = self.crypto_models[idx]
                try:
                    crypto_account = self.get_crypto_account(crypto_model.symbol)
                    usdc_account = self.get_crypto_account('USDC')
                except Account.DoesNotExist:
                    continue
                crypto_value = crypto_model.objects.using(self.database).latest('timestamp').close
                crypto_account.value += individual_liquidity/crypto_value
                crypto_account.save(using=self.database)
                usdc_account.value = 0
                usdc_account.save(using=self.database)
        # perform sell actions on DB
        if len(sell_indices) > 0:
            for idx in sell_indices:
                    crypto_model = self.crypto_models[idx]
                    try:
                        crypto_account = self.get_crypto_account(crypto_model.symbol)
                        usdc_account = self.get_crypto_account('USDC')
                    except Account.DoesNotExist:
                        continue
                    crypto_value = crypto_model.objects.using(self.database).latest('timestamp').close
                    total_value = crypto_value * crypto_account.value
                    usdc_account.value += total_value
                    usdc_account.save(using=self.database)
                    crypto_account.value = 0
                    crypto_account.save(using=self.database)
        new_timestamp = self.timestamp + timedelta(hours=1)
        done = False
        # fetching new crypto data
        for crypto in self.crypto_models:
            try:
                historical_data = crypto.objects.using(Database.HISTORICAL.value).filter(timestamp=new_timestamp).first()
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
        # self.volume_decay(factor=0.0005)
        self.step_count += 1
        info = {}
        return next_state, cost_for_action, done, info
    
    def volume_decay(self, factor:float=0.002) -> None:
        for crypto_model in crypto_models:
            try:
                crypto_account = self.get_crypto_account(crypto_model.symbol)
                crypto_account.value = crypto_account.value * (1 - factor)
                crypto_account.save()
            except Account.DoesNotExist:
                continue
        try:
            usdc_account = self.get_crypto_account('USDC')
            usdc_account.value = usdc_account.value * (1 - factor)
            usdc_account.save()
        except Account.DoesNotExist:
            return

    def get_crypto_account(self, symbol: str) -> Account:
        try:
            crypto_account = Account.objects.using(self.database).get(name=f'{symbol} Wallet')
            return crypto_account
        except Account.DoesNotExist:
            print(f'Account {symbol} Wallet does not exist!')
            raise Account.DoesNotExist

    def reset_state(self) -> npt.NDArray[np.float16]:
        self.step_count = 0
        self.initial_timestamp = None
        self.maker_fee = 0.004  # 0.4%
        self.taker_fee = 0.006  # 0.6%
        self.initial_volume = self.initial_volume
        self.database = Database.SIMULATION.value
        self.crypto_models:List[AbstractOHLCV] = crypto_models
        self.total_volume = self.initial_volume
        self.account_holdings = [0 for _ in self.crypto_models]
        self.earliest_timestamp = self.get_earliest_timestamp()
        self.maximum_timestamp = self.get_maximum_timestamp()
        self.buffer = 1024
        self.initial_timestamp = self.get_starting_timestamp()
        self.timestamp = self.initial_timestamp
        self.initial_prices = self.get_initial_crypto_prices()
        # self.timestamp = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0) #! this is the start time
        self.prediction_handler = PredictionHandler(lstm_sequence_length=100, database=self.database, timestamp=self.timestamp)
        self.prepare_simulation_database()
        initial_state = self.get_current_state()
        return initial_state
    
    def cost_for_action(self, action: Actions) -> float:
        total_cost = 0
        for idx, crypto_action in enumerate(action):
            if crypto_action == Actions.HOLD.value:
                continue
            is_buy = True if crypto_action == Actions.BUY.value else False
            crypto = self.crypto_models[idx]
            try:
                transaction_volume = self.calculate_transaction_volume(crypto, is_buy)
            except Account.DoesNotExist:
                continue
            fee_rate = self.maker_fee if is_buy else self.taker_fee  # Assuming maker fee for buy, taker fee for sell
            transaction_cost = transaction_volume * fee_rate
            total_cost += transaction_cost
        return max(total_cost, 0.0)
    
    def calculate_transaction_volume(self, crypto: AbstractOHLCV, is_buy: bool) -> float:
        if is_buy:
            crypto_account = self.get_crypto_account('USDC')
            transaction_volume = crypto_account.value
        else:
            crypto_account = self.get_crypto_account(crypto.symbol)
            price = crypto.objects.using(self.database).latest('timestamp').close
            transaction_volume = price * crypto_account.value
        return transaction_volume  # Assuming this is the result of your method

    def get_crypto_features(self) -> List[str]:
        return crypto_features

    def get_crypto_predicted_features(self) -> List[str]:
        return crypto_predicted_features
    
    def crypto_to_list(self, crypto: AbstractOHLCV) -> List[float]:
        return [getattr(crypto, fieldname) for fieldname in self.get_crypto_features()]
    
    def get_new_prediction_data(self, crypto_model:AbstractOHLCV, timestamp:datetime) -> List[float]:
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
    
    def prepare_simulation_database(self) -> None:
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

    def reset_account_data(self) -> None:
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
    
    def get_liquidity(self) -> float:
        try:
            usdc_account = self.get_crypto_account('USDC')
            return usdc_account.value
        except Account.DoesNotExist:
            print('USDC Account does not exist?!?')
            return 0
        
    def get_account_holdings(self) -> List[float]:
        account_holdings = []
        for crypto in self.crypto_models:
            try:
                account = Account.objects.using(self.database).get(name=f'{crypto.symbol} Wallet')
            except Account.DoesNotExist:
                print(f'Account "{crypto.symbol} Wallet" does not exist')
                continue
            account_holdings.append(account.value)
        return account_holdings
    
    def get_new_crypto_data(self) -> List[float]:
        all_entries = []
        # print('getting new crypto data:')
        for crypto in self.crypto_models:
            crypto_latest = crypto.objects.using(self.database).latest('timestamp')
            all_entries = all_entries + self.crypto_to_list(crypto_latest)
            all_entries = all_entries + self.get_new_prediction_data(crypto, crypto_latest.timestamp)
            # print('finished getting crypto data')
        # print(f'all entries: {all_entries}')
        return all_entries


class CustomEnv(gym.Env):
    def __init__(self, data_handler:AbstractDataHandler, asymmetry_factor:float=1, total_steps:int = 1024) -> None:
        # print('Initializing env')
        super(CustomEnv, self).__init__()
        self.step_count = 0
        self.total_steps = total_steps
        self.crypto_models = crypto_models
        self.data_handler = data_handler
        N = len(self.crypto_models)
        self.action_space = spaces.MultiDiscrete([3] * N)  # where N is the number of cryptocurrencies
        self.prev_total_volume = None
        self.prev_reward = None
        self.asymmetry_factor = asymmetry_factor
        self.volume_timeframe= 24*3
        self.volume_values = [0]*self.volume_timeframe
        self.fading_coefficient = 0.1
        # self.fading_coefficient = 0.75
        self.volume_coefficient = 1.0
        M = len(self.get_crypto_features()) + len(self.get_crypto_predicted_features()) + len(self.get_extra_features())
        shape_value = M*N + 2 #! +1 because of total volume held and USDC value held
        # print(f'M: {M}\tN: {N}\tM*N: {M*N}\n')
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape_value,), dtype=np.float64)

    def step(self, action: Actions) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        # print(f'stepping env with action: {action}')
        next_state, cost_for_action, terminated, info = self.data_handler.update_state(action)
        total_volume = next_state[0]
        self.volume_values = self.volume_values[1:] + [total_volume]
        usdc_held = next_state[1]
        reward_q = self.calculate_reward_volume_normalized(action, total_volume, cost_for_action)
        # reward_q = self.calculate_reward_quadratic(action, total_volume, cost_for_action)
        self.prev_total_volume = total_volume
        self.prev_reward = reward_q
        truncated = False
        reward_ratios = self.data_handler.get_reward_ratios_for_current_timestep()
        reward_string = f'gain:{total_volume/1000:.3f},'
        for key in reward_ratios:
            reward_string = reward_string + f'{key}:{reward_ratios[key]:.3f},'
        print(f'{self.data_handler.step_count}/{self.total_steps}: time: {self.data_handler.timestamp}. current volume: {total_volume:.2f}, usdc: {usdc_held:.2f}, {reward_string} reward {reward_q:.2f}, cost: {cost_for_action:.2f}')
        return next_state, reward_q, terminated, truncated, info
    
    def reset(self, seed=None, options=None) -> Tuple[npt.NDArray[np.float16], Dict[Any, Any]]:
        self.crypto_models = crypto_models
        self.prev_total_volume = None
        self.prev_reward = None
        self.volume_timeframe= 24*3
        self.volume_values = [0]*self.volume_timeframe
        initial_state = self.data_handler.reset_state()
        info = {}
        return initial_state, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_crypto_features(self) -> List[str]:
        return crypto_features

    def get_crypto_predicted_features(self) -> List[str]:
        return crypto_predicted_features
    
    def get_extra_features(self) -> List[str]:
        return crypto_extra_features

    def calculate_reward_quadratic(self, action: Actions, total_volume:float, cost_for_action:float) -> float:
        #todo redo quadratic, made linear for now
        if self.prev_total_volume is None:
            return 0
        volume_diff = total_volume - self.prev_total_volume
        if volume_diff > 0:
            reward = volume_diff
        else:
            reward = -self.asymmetry_factor * (volume_diff)
        reward = reward - cost_for_action
        return reward
    
    def calculate_reward_exponential(self, action: Actions, total_volume: float, cost_for_action: float) -> float:
        if self.prev_total_volume is None:
            return 0
        volume_diff = total_volume - self.prev_total_volume
        if volume_diff > 0:
            reward = np.exp(volume_diff) - 1
        else:
            reward = -self.asymmetry_factor * (np.exp(-volume_diff) - 1)
            reward = reward - cost_for_action
        return reward
    
    def calculate_reward_volume_normalized(self, action: Actions, total_volume: float, cost_for_action: float)-> float:
        if self.prev_total_volume is None or self.prev_reward is None:
            return 0
        # volume_diff = total_volume - self.prev_total_volume
        # std_dev = self.exponential_moving_std_dev()
        # normalized_diff = volume_diff / (std_dev + 1e-8)  # Adding a small value to avoid division by zero
        # norm_diff_string = ''
        # if normalized_diff > 0:
        #     reward = normalized_diff
        #     norm_diff_string = str(normalized_diff)
        # else:
        #     reward = self.asymmetry_factor * (normalized_diff)
        #     norm_diff_string = str(self.asymmetry_factor * (normalized_diff))
        
        # reward = self.fading_coefficient*self.prev_reward# + self.volume_coefficient*(np.log(total_volume / self.data_handler.initial_volume))
        # reward = reward + self.fading_coefficient*self.prev_reward + self.volume_coefficient*(np.log(total_volume / self.data_handler.initial_volume))
        # reward = reward - cost_for_action
        # reward_ratios = self.data_handler.get_reward_ratios_for_current_timestep()
        # max_reward_ratio = max([item for item in reward_ratios.values()])
        # avg_reward_ratio = sum([item for item in reward_ratios.values()])/len(reward_ratios)
        reward = ((total_volume / self.data_handler.initial_volume) - 1)*3
        # reward = reward + ((total_volume / self.data_handler.initial_volume)/max_reward_ratio - 1)*3
        # print(f"reward: {reward:.2f} = {norm_diff_string:.2f} + {self.fading_coefficient*self.prev_reward:.2f} + {self.volume_coefficient*(np.log(total_volume / self.data_handler.initial_volume)):.2f} - {cost_for_action:.2f} + {(total_volume / self.data_handler.initial_volume)/max_reward_ratio - 1:.2f}")
        # reward = reward*(1-self.fading_coefficient+0.01)*2.5
        return reward
    
    def calculate_reward_sharpe_ratio(self,action: Actions, total_volume: float, cost_for_action: float) -> float:
        #TODO finish implementing the submethods for this method
        expected_return = self.get_expected_return()
        std_dev_returns = self.get_std_dev_returns()
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
    
    def exponential_moving_std_dev(self) -> float:
        ema = self.calculate_ema(self.volume_values, self.volume_timeframe)
        squared_diff = [(val - ema[i]) ** 2 for i, val in enumerate(self.volume_values)]
        emsd_values = self.calculate_ema(squared_diff, self.volume_timeframe)
        return emsd_values[-1] ** 0.5  # Square root of the most recent EMSD value
    
    def get_expected_return(self):
        #TODO implement this method
        # expected_return = self.data['daily_return'].mean()
        return 0
    
    def get_std_dev_returns(self):
        #TODO implement this method
        return 0

    