from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
import numpy.typing as npt
from coinbase_api.models.models import AbstractOHLCV, Account, Prediction, CryptoMetadata
from coinbase_api.constants import crypto_models, crypto_features, crypto_predicted_features
from coinbase_api.enums import Actions, Database
from datetime import datetime, timedelta
import random
import time
import logging

class SimulationDataHandler:
    def __init__(self, initial_volume=1000, total_steps=1024, transaction_cost_factor=1.0, reward_function_index=0) -> None:
        logging.basicConfig(
            filename='simulation.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s'
        )
        self.timestamp_to_index = {}
        self.timing_count = 0
        self.logger = logging.getLogger(__name__)
        self.reward_function_index = reward_function_index
        reward_function_index_max = 2
        if (reward_function_index > reward_function_index_max):
            self.logger.critical(f"Reward function index {reward_function_index} is bigger than the maximum {reward_function_index_max}. using 0 instead!")
            self.reward_function_index = 0
        self.initial_timestamp = None
        self.lstm_sequence_length = 100
        self.total_steps = total_steps
        self.transaction_cost_factor = transaction_cost_factor
        self.action_factor = 0.5
        self.step_count = 0
        self.feature_indices = [i for i, feature in enumerate(['timestamp', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns']) if feature in crypto_features]
        self.usdc_held = initial_volume  # Initialize USDC account in memory
        self.minimum_number_of_cryptos = 100
        self.maker_fee = 0.004  # 0.4%
        self.taker_fee = 0.006  # 0.6%
        self.initial_volume: float = initial_volume
        self.database: Database = Database.SIMULATION.value
        self.crypto_models: List[AbstractOHLCV] = crypto_models
        self.symbols = [crypto.symbol for crypto in self.crypto_models]
        self.total_volume = initial_volume
        self.account_holdings = {crypto.symbol: 0.0 for crypto in self.crypto_models}
        
        self.earliest_timestamp = self.get_earliest_timestamp()
        self.maximum_timestamp = self.get_maximum_timestamp()
        self.buffer = 2*total_steps
        self.initial_timestamp = self.get_starting_timestamp()
        self.timestamp = self.initial_timestamp
        self.dataframe_cache = self.fetch_all_historical_data()
        self.prediction_cache = self.fetch_all_prediction_data()
        # self.symbol_indices = np.array([self.symbol_to_index[symbol] for symbol in self.symbols])
        self.state = self.get_current_state()
        self.past_volumes = []
        self.short_term_reward_window = 10
        self._initialized = True

    def fetch_all_historical_data(self) -> Dict[str, pd.DataFrame]:
        dataframes = {}
        start_timestamp = self.timestamp - timedelta(hours=self.lstm_sequence_length - 1)
        end_timestamp = self.timestamp + timedelta(hours=self.total_steps)
        complete_index = pd.date_range(start=start_timestamp, end=end_timestamp, freq='H')
        columns = ['timestamp', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns']

        for crypto_model in self.crypto_models:
            data = crypto_model.objects.using(Database.HISTORICAL.value).filter(
                timestamp__range=(start_timestamp, end_timestamp)
            ).values(
                'timestamp', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns'
            ).order_by('timestamp')

            df = pd.DataFrame(data)
            if 'timestamp' not in df.columns or df.empty:
                df = pd.DataFrame(columns=columns)
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df[~df.index.duplicated(keep='first')]

            df = df.reindex(complete_index)
            df['symbol'] = crypto_model.symbol
            df['symbol'].fillna(crypto_model.symbol, inplace=True)
            df.fillna(0, inplace=True)
            # Ensure no negative or zero values for critical columns
            for column in ['close', 'volume']:
                df[column] = df[column].apply(lambda x: max(x, 0))
            dataframes[crypto_model.symbol] = df
            if not self.timestamp_to_index:
                self.timestamp_to_index = {timestamp: idx for idx, timestamp in enumerate(df.index)}
        self.dataframe_cache_np = self.convert_dataframe_cache_to_combined_numpy(dataframes)
        return dataframes
    
    def convert_dataframe_cache_to_combined_numpy(self, dataframes: Dict[str, pd.DataFrame]) -> np.ndarray:
        timestamps = dataframes[next(iter(dataframes))].index
        num_features = len(crypto_features)
        num_symbols = len(self.symbols)
        
        # Initialize the combined array
        combined_data = np.zeros((len(timestamps), num_symbols * num_features))
        for i, symbol in enumerate(self.symbols):
            combined_data[:, i * num_features:(i + 1) * num_features] = dataframes[symbol][crypto_features].to_numpy()
        return combined_data


    def fetch_all_prediction_data(self) -> Dict[str, Dict[datetime, List[float]]]:
        start_time = time.time()
        # Initialize the dictionary for storing prediction data
        prediction_data = {crypto.symbol: {} for crypto in self.crypto_models}
        # Define the time range for fetching predictions
        start_timestamp = self.timestamp - timedelta(hours=self.lstm_sequence_length + 1)
        end_timestamp = self.timestamp + timedelta(hours=self.total_steps)
        # Fetch predictions from the database
        predictions = list(Prediction.objects.using(Database.HISTORICAL.value).filter(
            timestamp_predicted_for__range=(start_timestamp, end_timestamp),
            crypto__in=[crypto.symbol for crypto in self.crypto_models]
        ).values(
            'timestamp_predicted_for', 'crypto', 'model_name', 'predicted_field', 'predicted_value'
        ).iterator())
        # Process the fetched predictions and insert them into the dictionary
        for pred in predictions:
            symbol = pred['crypto']
            timestamp = pred['timestamp_predicted_for']
            index = 0 if pred['predicted_field'] == 'close_higher_shifted_1h' else 1 if pred['predicted_field'] == 'close_higher_shifted_24h' else 2
            index += 0 if pred['model_name'] == 'LSTM' else 3
            if timestamp not in prediction_data[symbol]:
                prediction_data[symbol][timestamp] = [0.0] * 6  # 3 for LSTM and 3 for XGBoost
            prediction_data[symbol][timestamp][index] = pred['predicted_value']
        end_time = time.time() - start_time
        print(f'time to fetch all predictions: {end_time:.2f} = {len(crypto_models)} * {end_time / len(crypto_models):.3f}')
        return prediction_data

    def get_earliest_timestamp(self) -> datetime:
        all_information = CryptoMetadata.objects.using(Database.HISTORICAL.value).all()
        timestamps = all_information.values_list("earliest_date").order_by("earliest_date")
        return timestamps[self.minimum_number_of_cryptos][0]

    def get_maximum_timestamp(self) -> datetime:
        all_timestamps = []
        for crypto_model in self.crypto_models:
            try:
                val = crypto_model.objects.using(Database.HISTORICAL.value).values_list("timestamp").order_by("-timestamp").first()
                all_timestamps.append(val if val is not None else (datetime(year=2024, month=6, day=20),))
            except crypto_model.DoesNotExist:
                print(f'did not exist: {crypto_model.symbol}')
                all_timestamps.append(crypto_model.default_entry(timestamp=datetime(year=2024, month=6, day=20)))
            except AttributeError:
                print(f'attribute error: {crypto_model.symbol}')
                all_timestamps.append(crypto_model.default_entry(timestamp=datetime(year=2024, month=6, day=20)))
        all_timestamps.sort()
        return all_timestamps[0][0]

    def get_starting_timestamp(self) -> datetime:
        if self.initial_timestamp is None:
            hours = self.maximum_timestamp - self.earliest_timestamp
            hours_number = hours.total_seconds() // 3600
            rand_start = random.randint(0, int(hours_number) - self.buffer)
            print(f'starting timestamp: {self.earliest_timestamp + timedelta(hours=rand_start)}\tminimum: {self.earliest_timestamp}\tmaximum: {self.maximum_timestamp}\t random number: {rand_start}/{int(hours_number)}')
            return self.earliest_timestamp + timedelta(hours=rand_start)
        print('initial timestamp was already set')
        return self.initial_timestamp

    def get_current_state(self) -> npt.NDArray[np.float16]:
        prices = np.array([self.get_crypto_value_from_cache(symbol, self.timestamp) for symbol in self.account_holdings.keys()])
        holdings = np.array(list(self.account_holdings.values()))
        self.total_volume = np.sum(holdings * prices) + self.usdc_held
        
        if self.step_count >= self.total_steps - 1:
            print(f"finished training with: {self.total_volume:.2f}")
        self.new_crypto_data = self.get_new_crypto_data()
        state = np.array([self.total_volume, self.usdc_held] + list(self.account_holdings.values()) + self.new_crypto_data)
        return state

    def map_buy_action(self, buy_action: float, action_factor: float) -> float:
        return (buy_action - action_factor) / (1 - action_factor)

    def map_sell_action(self, sell_action: float, action_factor: float) -> float:
        return (max(sell_action + action_factor - 0.1, action_factor - 1)) / (1 - action_factor)

    def update_state(self, action) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        self.costs_for_action = self.cost_for_action(action[1:])  # Exclude the first action for costs calculation
        
        buy_indices = [(idx, i) for idx, i in enumerate(action[1:]) if i > self.action_factor]
        sell_indices = [(idx, i) for idx, i in enumerate(action[1:]) if i < -self.action_factor]
        hold_usdc = action[0] < 0  # First action is for holding USDC
        available_liquidity = self.usdc_held - sum(self.costs_for_action)
        N = 5
        top_buy_indices = sorted(buy_indices, key=lambda x: -x[1])[:N]
        if not hold_usdc:
            total_buy_actions = sum([self.map_buy_action(buy_action, self.action_factor) for idx, buy_action in top_buy_indices])
            individual_liquidity = available_liquidity / total_buy_actions if total_buy_actions > 0 else 0
            for idx, buy_action in top_buy_indices:
                crypto_model = self.crypto_models[idx]
                buy_action_mapped = self.map_buy_action(buy_action, self.action_factor)
                crypto_value = self.get_crypto_value_from_cache(crypto_model.symbol, self.timestamp)
                buy_amount = individual_liquidity * buy_action_mapped if crypto_value else 0
                self.account_holdings[crypto_model.symbol] = max(self.account_holdings[crypto_model.symbol] + (buy_amount - self.costs_for_action[idx]) / crypto_value if crypto_value else self.account_holdings[crypto_model.symbol], 0)
                self.usdc_held = max(self.usdc_held - buy_amount, 0)
        for idx, sell_action in sell_indices:
            crypto_model = self.crypto_models[idx]
            sell_action_mapped = self.map_sell_action(sell_action, self.action_factor)
            crypto_value = self.get_crypto_value_from_cache(crypto_model.symbol, self.timestamp)
            sell_amount = abs(self.account_holdings[crypto_model.symbol] * sell_action_mapped) if crypto_value else 0
            self.usdc_held = max(self.usdc_held + sell_amount * crypto_value - self.costs_for_action[idx] if crypto_value else self.usdc_held, 0)
            self.account_holdings[crypto_model.symbol] = max(self.account_holdings[crypto_model.symbol] - sell_amount, 0)
        self.timestamp += timedelta(hours=1)
        self.state = self.get_current_state()
        self.past_volumes.append(self.total_volume)
        if len(self.past_volumes) > self.short_term_reward_window:
            self.past_volumes.pop(0)
        self.step_count += 1
        done = False
        info = {}
        return self.state, sum(self.costs_for_action), done, info

    def get_crypto_value_from_cache(self, symbol: str, timestamp: datetime) -> float:
        return self.dataframe_cache[symbol].at[timestamp, 'close']

    def get_crypto_data_from_cache(self, symbol: str, timestamp: datetime) -> List[float]:
        df = self.dataframe_cache[symbol]
        if timestamp in df.index:
            row = df.loc[timestamp]
            return [
                row[feature] for feature in crypto_features
            ]
        return [0] * 10  # Return default values if the timestamp is not found
    
    def get_crypto_values_from_cache_vectorized(self, timestamp: datetime) -> np.ndarray:
        return np.array([self.dataframe_cache[crypto.symbol].at[timestamp, 'close'] for crypto in self.crypto_models])

    def reset_state(self) -> npt.NDArray[np.float16]:
        self.logger.info("Resetting state")
        # Log initial state information
        self.logger.info(f"Total steps: {self.total_steps}")
        self.logger.info(f"Initial timestamp: {self.initial_timestamp}")
        self.logger.info(f"Current timestamp: {self.timestamp}")
        self.logger.info(f"Step number: {self.step_count}")
        self.logger.info(f"USDC held before reset: {self.usdc_held}")
        self.logger.info(f"Crypto values before reset: {self.account_holdings}")
        self.step_count = 0
        self.action_factor = 0.2
        self.initial_timestamp = None
        self.maker_fee = 0.004  # 0.4%
        self.taker_fee = 0.006  # 0.6%
        self.initial_volume = self.initial_volume
        self.database = Database.SIMULATION.value
        self.crypto_models: List[AbstractOHLCV] = crypto_models
        self.total_volume = self.initial_volume
        self.account_holdings = {crypto.symbol: 0 for crypto in self.crypto_models}
        self.symbols = [crypto.symbol for crypto in self.crypto_models]
        self.feature_indices = [i for i, feature in enumerate(['timestamp', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'bollinger_high', 'bollinger_low', 'vmap', 'percentage_returns', 'log_returns']) if feature in crypto_features]
        
        if not hasattr(self, '_initialized'):
            self.timestamp_to_index = {}
            self.earliest_timestamp = self.get_earliest_timestamp()
            self.maximum_timestamp = self.get_maximum_timestamp()
            self.buffer = 2*self.total_steps
            self.initial_timestamp = self.get_starting_timestamp()
            self.timestamp = self.initial_timestamp
            self.dataframe_cache = self.fetch_all_historical_data()
            # self.symbol_indices = np.array([self.symbol_to_index[symbol] for symbol in self.symbols])
            self.prediction_cache = self.fetch_all_prediction_data()
            self.prepare_simulation_database()
        else:
            if (self._initialized):
                self._initialized = False
            else:
                self.timestamp_to_index = {}
                self.earliest_timestamp = self.get_earliest_timestamp()
                self.maximum_timestamp = self.get_maximum_timestamp()
                self.buffer = 2*self.total_steps
                self.initial_timestamp = self.get_starting_timestamp()
                self.timestamp = self.initial_timestamp
                self.dataframe_cache = self.fetch_all_historical_data()
                self.prediction_cache = self.fetch_all_prediction_data()
                self.prepare_simulation_database()
                
        self.initial_state = self.get_current_state()
        return self.initial_state
    
    def map_buy_action_array(self, buy_actions: np.ndarray, action_factor: float) -> np.ndarray:
        return np.maximum(buy_actions - action_factor, 0) / (1 - action_factor)

    def map_sell_action_array(self, sell_actions: np.ndarray, action_factor: float) -> np.ndarray:
        return np.maximum(sell_actions + action_factor - 0.1, action_factor - 1) / (1 - action_factor)
    
    def cost_for_action(self, action: List[float]) -> List[float]:
        action_array = np.array(action)
        buy_actions = action_array >= self.action_factor
        sell_actions = action_array < -self.action_factor

        mapped_buy_actions = np.maximum(self.map_buy_action_array(action_array, self.action_factor), 0)
        total_buy_action = np.sum(mapped_buy_actions[buy_actions])

        if total_buy_action == 0:
            total_buy_action = 1
            
        # Fetch current crypto values from cache
        prices = self.get_crypto_values_from_cache_vectorized(self.timestamp)
        
        # Calculate transaction volumes
        buy_proportions = mapped_buy_actions / total_buy_action
        transaction_volumes = np.where(
            buy_actions,
            self.usdc_held * buy_proportions,
            np.where(
                sell_actions,
                np.abs(np.array(list(self.account_holdings.values())) * self.map_sell_action_array(action_array, self.action_factor)) * prices,
                0
            )
        )

        fee_rates = np.where(buy_actions, self.maker_fee, np.where(sell_actions, self.taker_fee, 0))
        transaction_costs = transaction_volumes * fee_rates * self.transaction_cost_factor
        return transaction_costs.tolist()

    def get_new_prediction_data(self, timestamp: datetime) -> Dict[str, List[float]]:
        all_entries = {
            crypto.symbol: self.prediction_cache[crypto.symbol].get(timestamp, [0.0] * 6)
            for crypto in self.crypto_models
        }
        return all_entries

    def prepare_simulation_database(self) -> None:
        print(f'Preparing simulation.')
        self.reset_account_data()

    def reset_account_data(self) -> None:
        self.usdc_held = self.initial_volume  # Initialize USDC in memory
    
    def get_timestamp_index(self, timestamp: datetime) -> int:
        return self.timestamp_to_index.get(timestamp, -1)
        
    def get_new_crypto_data(self) -> List[float]:
        prediction_data = self.get_new_prediction_data(self.timestamp)
        timestamp_idx = self.get_timestamp_index(self.timestamp)
        if timestamp_idx == -1:
            raise ValueError(f"Timestamp {self.timestamp} not found in index")
        crypto_data_matrix = self.dataframe_cache_np[timestamp_idx]
        all_entries = crypto_data_matrix.tolist()
        for crypto in self.crypto_models:
            all_entries.extend(prediction_data[crypto.symbol])
        return all_entries

    def get_reward(self, action: Actions) -> float:
        if self.reward_function_index == 0:
            return self.reward_function_0(action)
        elif self.reward_function_index == 1:
            return self.reward_function_1(action)
        elif self.reward_function_index == 2:
            return self.reward_function_2(action)
        elif self.reward_function_index == 3:
            return self.reward_function_3(action)
        else:
            return 0.0

    def reward_function_0(self, action: Actions) -> float:
        net_return = (self.total_volume / self.initial_volume) - 1
        reward = net_return * 1.0  # Scale the reward as needed
        asymmetry_factor = 1.0  # Adjust this factor to control asymmetry

        cumulative_reward = 0
        if len(self.past_volumes) >= 1:
            past_volume = np.sum(self.past_volumes) / len(self.past_volumes)
            short_term_gain = (self.total_volume - past_volume) / past_volume
            cumulative_reward += short_term_gain * 5
        
        reward = net_return * 0.25 + cumulative_reward * 1.0
        reward *= 5
        if reward > 0:
            reward *= asymmetry_factor
        return reward

    def reward_function_1(self, action: Actions) -> float:
        net_return = (self.total_volume / self.initial_volume) - 1
        reward = net_return * 1.0  # Scale the reward as needed
        asymmetry_factor = 1.0  # Adjust this factor to control asymmetry

        cumulative_reward = 0
        if len(self.past_volumes) >= 1:
            past_volume = np.sum(self.past_volumes) / len(self.past_volumes)
            short_term_gain = (self.total_volume - past_volume) / past_volume
            cumulative_reward += short_term_gain * 5

        transaction_costs = np.sum(self.costs_for_action)
        
        reward = net_return * 0.25 + cumulative_reward * 1.0 - transaction_costs * 0.5
        reward *= 5
        if reward > 0:
            reward *= asymmetry_factor
        return reward

    def reward_function_2(self, action: Actions) -> float:
        net_return = (self.total_volume / self.initial_volume) - 1
        reward = net_return * 1.0  # Scale the reward as needed
        asymmetry_factor = 1.0  # Adjust this factor to control asymmetry

        cumulative_reward = 0
        if len(self.past_volumes) >= 1:
            past_volume = np.sum(self.past_volumes) / len(self.past_volumes)
            short_term_gain = (self.total_volume - past_volume) / past_volume
            cumulative_reward += short_term_gain * 5
        
        transaction_costs = np.sum(self.costs_for_action)
        max_drawdown = np.min(self.total_volume / np.maximum.accumulate(self.past_volumes)) - 1
        drawdown_penalty = max_drawdown * 0.5
        
        reward = net_return * 0.25 + cumulative_reward * 1.0 - transaction_costs * 0.25 - drawdown_penalty
        reward *= 5
        if reward > 0:
            reward *= asymmetry_factor
        return reward
    
    def reward_function_3(self, action: Actions) -> float:
        net_return = (self.total_volume / self.initial_volume) - 1
        reward = net_return * 1.0  # Scale the reward as needed
        asymmetry_factor = 1.0  # Adjust this factor to control asymmetry

        cumulative_reward = 0
        if len(self.past_volumes) >= 1:
            past_volume = np.sum(self.past_volumes) / len(self.past_volumes)
            short_term_gain = (self.total_volume - past_volume) / past_volume
            cumulative_reward += short_term_gain * 5
        
        transaction_costs = np.sum(self.costs_for_action)
        
        # Penalize if holding USDT is less than 0
        holding_penalty = 0
        if action[0] > 0:
            holding_penalty = 5  # Adjust the penalty factor as needed

        reward = net_return * 0.25 + cumulative_reward * 1.0 - transaction_costs * 0.5 - holding_penalty
        reward *= 5
        if reward > 0:
            reward *= asymmetry_factor
        return reward
