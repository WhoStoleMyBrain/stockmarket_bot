import os
from typing import Any, Dict, List, Tuple
import numpy as np
import numpy.typing as npt
from coinbase_api.models.models import AbstractOHLCV, BracketSellItem, CryptoMetadata
from coinbase_api.constants import crypto_features, simulation_columns
from coinbase_api.enums import Actions, Database, ExportFolder
from datetime import datetime, timedelta
import random
import logging
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
try:
    import pyarrow.dataset as ds
except ImportError:
    ds = None  # Fall back to in-memory filtering if PyArrow is not available

class SimulationDataHandler:
    def __init__(self, crypto: AbstractOHLCV, initial_volume=1000, total_steps=1024, transaction_cost_factor=1.0, reward_function_index=0, noise_level=0.01, slippage_level=0.00) -> None:
        logging.basicConfig(
            filename='/logs/sim_logs/simulation.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s'
        )
        self.crypto = crypto
        self.timestamp_to_index = {}
        self.holding_actions = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.bracketSells:list[BracketSellItem] = []
        self.timing_count = 0
        self.noise_level = noise_level
        self.slippage_level = slippage_level
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
        self.minimum_number_of_cryptos = 1 #? set to 1 because we will start handling data in sequence instead of parallel
        self.maker_fee = 0.0025  # 0.25% based on Advanced 1, might drop
        self.taker_fee = 0.004  # 0.4% based on Advanced 1, might drop
        self.initial_volume: float = initial_volume
        self.total_volume = initial_volume
        self.account_holdings = {self.crypto.symbol: 0.0}
        
        self.earliest_timestamp = self.get_earliest_timestamp()
        self.maximum_timestamp = self.get_maximum_timestamp()
        self.buffer = 2*total_steps
        self.initial_timestamp = self.get_starting_timestamp()
        self.timestamp = self.initial_timestamp
        # self.dataframe_cache = self.fetch_all_historical_data()
        self.dataframe_cache = self.fetch_all_historical_data_from_file()
        self.dataframe_cache_np = self.convert_dataframe_cache_to_combined_numpy(self.dataframe_cache)
        
        # self.prediction_cache = self.fetch_all_prediction_data()
        self.state = self.get_current_state()
        self.past_volumes = []
        self.short_term_reward_window = 10
        # self.logger.setLevel(logging.WARNING)
        self._initialized = True
        
    def fetch_all_historical_data_from_file(self) -> dict:
        """
        Load historical data for the current crypto from a Parquet file.
        If the file does not exist, an exception is raised (or you could fall back to a database query).
        """
        #! Noise?
        file_path = os.path.join(ExportFolder.EXPORT_FOLDER.value, f"{self.crypto.symbol}.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Historical data file for {self.crypto.symbol} not found at {file_path}")
        
        # Compute the start and end timestamps for the data needed.
        start_timestamp = self.timestamp - timedelta(minutes=(self.lstm_sequence_length - 1) * 5)
        end_timestamp = self.timestamp + timedelta(minutes=self.total_steps * 5)
        # Read the Parquet file into a DataFrame.
        # Option 1: Use PyArrow Dataset filtering if available.
        if ds is not None:
            dataset = ds.dataset(file_path, format="parquet")
            # Filter rows using the 'timestamp' column; note that the column name must match what was written.
            filtered_table = dataset.to_table(filter=(
                (ds.field("timestamp") >= start_timestamp) & (ds.field("timestamp") <= end_timestamp)
            ))
            df = filtered_table.to_pandas()
        else:
            # Option 2: Fall back to reading the full file and filtering in-memory.
            df = pd.read_parquet(file_path)
            # If the file’s index is not already datetime, convert the 'timestamp' column.
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            df = df.loc[start_timestamp:end_timestamp]
        # Ensure the index is datetime and sorted.
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        # **NEW**: Fill missing values so no NaNs propagate.
        df.fillna(0, inplace=True)
        # Return a dictionary mapping symbol to its DataFrame (for compatibility with your code).
        self.timestamp_to_index = {ts: idx for idx, ts in enumerate(df.index)}
        # print(df.head(100))
        return {self.crypto.symbol: df}

    def fetch_all_historical_data(self) -> Dict[str, pd.DataFrame]:
        dataframes = {}
        start_timestamp = self.timestamp - timedelta(minutes=(self.lstm_sequence_length - 1) * 5)
        end_timestamp = self.timestamp + timedelta(minutes=self.total_steps * 5)
        complete_index = pd.date_range(start=start_timestamp, end=end_timestamp, freq='5min')
        # columns = simulation_columns
        data = self.crypto.objects.using(Database.HISTORICAL.value).filter(
            timestamp__range=(start_timestamp, end_timestamp)
        ).values(
            *simulation_columns
        ).order_by('timestamp')

        df = pd.DataFrame(data)
        if 'timestamp' not in df.columns or df.empty:
            df = pd.DataFrame(columns=simulation_columns)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]

        df = df.reindex(complete_index)
        df['symbol'] = self.crypto.symbol
        df['symbol'].fillna(self.crypto.symbol, inplace=True)
        df.fillna(0, inplace=True)
        # Ensure no negative or zero values for critical columns
        for column in ['close', 'volume']:
            df[column] = df[column].apply(lambda x: max(x, 0))
        if (self.noise_level > 0): #! currently not implemented
            df = self.add_noise_to_dataframe(df, columns=['close', 'volume', 'sma', 'ema', 'rsi', 'macd', 
                                                'bollinger_high', 'bollinger_low', 'vmap', 
                                                'percentage_returns', 'log_returns'], noise_level=self.noise_level)
        dataframes[self.crypto.symbol] = df
        if not self.timestamp_to_index:
            self.timestamp_to_index = {timestamp: idx for idx, timestamp in enumerate(df.index)}
        self.dataframe_cache_np = self.convert_dataframe_cache_to_combined_numpy(dataframes)
        return dataframes
    
    def add_noise_to_dataframe(self, df: pd.DataFrame, columns: List[str], noise_level: float = 0.01) -> pd.DataFrame:
        """
        Add Gaussian noise to specific columns of the DataFrame.
        
        :param df: The DataFrame to which noise will be added.
        :param columns: List of column names where noise should be added.
        :param noise_level: The standard deviation of the Gaussian noise to be added.
        :return: DataFrame with noise added to the specified columns.
        """
        noise = np.random.normal(0, noise_level, df[columns].shape)
        df[columns] += df[columns] * noise
        return df
    
    def convert_dataframe_cache_to_combined_numpy(self, dataframes: dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Convert the cached DataFrame for the current crypto into a combined numpy array.
        The columns are ordered as specified by the global 'crypto_features' list.
        """
        # Get the DataFrame for the current crypto.
        df = dataframes[self.crypto.symbol]
        # Ensure the DataFrame has the required columns.
        if not all(col in df.columns for col in crypto_features):
            raise ValueError(f"DataFrame for {self.crypto.symbol} is missing one or more crypto_features")
        # Convert the specified columns to a numpy array.
        combined_data = df[crypto_features].to_numpy()
        # print(combined_data[:100])
        return combined_data

    def load_crypto_metadata(self, symbol: str) -> pd.Series:
        """
        Load the CryptoMetadata row for the given symbol from the Parquet file.
        The symbol is converted to its stored format using CryptoMetadata.symbol_to_storage.
        
        :param symbol: A string such as "BTC"
        :return: A pandas Series corresponding to the metadata row.
        :raises FileNotFoundError: If the metadata file does not exist.
        :raises ValueError: If no metadata entry is found for the symbol.
        """
        file_path = os.path.join(ExportFolder.EXPORT_FOLDER.value, "crypto_metadata.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metadata file not found at {file_path}")

        # Read the entire metadata file into memory.
        df = pd.read_parquet(file_path)
        # Convert the symbol to its stored version.
        stored_symbol = CryptoMetadata.symbol_to_storage(symbol)
        # Filter in-memory.
        row = df.loc[df["symbol"] == stored_symbol]
        if row.empty:
            raise ValueError(f"No metadata found for symbol {symbol}")
        # Return the first (and only) matching row as a pandas Series.
        return row.iloc[0]


    def get_earliest_timestamp(self) -> datetime:
        # Use the load_crypto_metadata helper to get the metadata row.
        metadata = self.load_crypto_metadata(self.crypto.symbol)
        return metadata["earliest_date"]
    
    def get_maximum_timestamp(self) -> datetime:
        """
        Read the Parquet file for the current crypto and return the newest timestamp
        by reading only metadata from each row group. If any error occurs or no valid
        timestamp is found, returns a default datetime.
        """
        file_path = os.path.join(ExportFolder.EXPORT_FOLDER.value, f"{self.crypto.symbol}.parquet")
        try:
            # Open the Parquet file.
            pf = pq.ParquetFile(file_path)
            # Use the available 'names' property from the ParquetSchema.
            schema_names = pf.schema.names
            if "timestamp" not in schema_names:
                raise ValueError("The 'timestamp' column is missing in the Parquet file.")
            timestamp_index = schema_names.index("timestamp")
            # Convert the Parquet schema to an Arrow schema.
            arrow_schema = pf.schema.to_arrow_schema()
            timestamp_field = arrow_schema.field(timestamp_index)
            unit = None
            if isinstance(timestamp_field.type, pa.TimestampType):
                unit = timestamp_field.type.unit  # e.g. "ms", "us", "ns"
            
            max_ts = None
            # Iterate over row groups to read the max statistic for the timestamp column.
            for rg in range(pf.num_row_groups):
                rg_meta = pf.metadata.row_group(rg)
                col_meta = rg_meta.column(timestamp_index)
                stats = col_meta.statistics
                if stats is None or stats.max is None:
                    continue
                current_max = stats.max
                try:
                    # Convert using the detected unit if available.
                    current_max_dt = pd.to_datetime(current_max, unit=unit) if unit else pd.to_datetime(current_max)
                except Exception:
                    current_max_dt = pd.to_datetime(current_max)
                if max_ts is None or current_max_dt > max_ts:
                    max_ts = current_max_dt
            
            if max_ts is None or pd.isnull(max_ts):
                raise ValueError("No valid timestamp found in file metadata.")
            # Return as a Python datetime.
            return max_ts.to_pydatetime() if hasattr(max_ts, "to_pydatetime") else max_ts
        except Exception as e:
            print(f"Error retrieving maximum timestamp for {self.crypto.symbol}: {e}")
            return datetime(year=2025, month=1, day=9)

    def get_starting_timestamp(self) -> datetime:
        if self.initial_timestamp is None:
            hours = self.maximum_timestamp - self.earliest_timestamp
            hours_number = hours.total_seconds() // 3600
            rand_start = random.randint(0, int(hours_number) - int((self.buffer/12))) #! /12 because we use 5 min intervals instead of 1 hour. Buffer is 2 * total_steps, so total_steps/6 in hourly segments
            print(f'starting timestamp: {self.earliest_timestamp + timedelta(hours=rand_start)}\tminimum: {self.earliest_timestamp}\tmaximum: {self.maximum_timestamp}\t random number: {rand_start}/{int(hours_number)}')
            return self.earliest_timestamp + timedelta(hours=rand_start)
        print('initial timestamp was already set')
        return self.initial_timestamp

    def get_current_state(self) -> npt.NDArray[np.float16]:
        crypto_price = self.get_crypto_value_from_cache(self.crypto.symbol, self.timestamp)
        holdings = np.array(list(self.account_holdings.values()))
        # bracketSellHoldings = sum([bracketSellItem.cryptoAmount * bracketSellItem.cryptoValue for bracketSellItem in self.bracketSells])
        self.total_volume = np.sum(holdings * crypto_price) + self.usdc_held
        # self.total_volume = np.sum(holdings * crypto_price) + self.usdc_held + bracketSellHoldings
        
        if self.step_count >= self.total_steps - 1 and self.total_volume != 0.0:
            expected_total_gain = 1.03**self.winning_trades * 0.99**self.losing_trades * 1000.0
            total_actions = self.winning_trades + self.losing_trades
            if total_actions != 0:
                winning_trade_percentage = self.winning_trades / total_actions
                losing_trade_percentage = self.losing_trades / total_actions
            else:
                winning_trade_percentage = 0
                losing_trade_percentage = 0
            total_days = len(self.holding_actions) / 288.0 # 1440/5 = 288
            actions_per_day = total_actions / float(total_days)
            self.logger.info(f"fin w/: {self.total_volume:.2f}$. est: {expected_total_gain:.2f}$. Hodl: {sum(self.holding_actions)} / {len(self.holding_actions)}. win: {self.winning_trades}({winning_trade_percentage*100:.1f}%). lose: {self.losing_trades}({losing_trade_percentage*100:.1f}%). Act/d: {actions_per_day:.2f}. timerange: {self.initial_timestamp} - {self.timestamp}")
        new_crypto_data = self.get_new_crypto_data()
        state = np.concatenate((np.array([self.total_volume, self.usdc_held]),
                                holdings, new_crypto_data))
        return state

    def map_buy_action(self, buy_action: float, action_factor: float) -> float:
        return (buy_action - action_factor) / (1 - action_factor)

    def map_sell_action(self, sell_action: float, action_factor: float) -> float:
        return (max(sell_action + action_factor - 0.1, action_factor - 1)) / (1 - action_factor)

    def apply_slippage_to_value(self, value: float, percentage: float = 2.0) -> float:
        rand_factor = 1.0 + random.uniform(0, percentage) / 100.0
        return value * rand_factor
    
    def apply_latency(self, percentage: float = 2.0) -> bool:
        return random.uniform(0, 100) < percentage
    
    def set_currency(self, new_currency: AbstractOHLCV, verbose=True):
        self.logger.info(f"setting crypto from {self.crypto.symbol} to {new_currency.symbol}")
        self.crypto = new_currency
        self.timestamp_to_index = {}
        self.account_holdings = {self.crypto.symbol: 0.0}
        self.earliest_timestamp = self.get_earliest_timestamp()
        self.maximum_timestamp = self.get_maximum_timestamp()
        self.initial_timestamp = None
        self.initial_timestamp = self.get_starting_timestamp()
        self.timestamp = self.initial_timestamp
        self.dataframe_cache = self.fetch_all_historical_data_from_file()
        self.state = self.get_current_state()

    def update_state(self, action) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        crypto_value = self.get_crypto_value_from_cache(self.crypto.symbol, self.timestamp)
        if not crypto_value:
            self.logger.warning(f"No crypto value found for {self.timestamp}")
            return self.state, 0.0, False, {}
        self.costs_for_action = self.cost_for_action(action)  # Exclude the first action for costs calculation. first action is usdc hold action
        is_buy = (action[0] > self.action_factor) and (self.usdc_held > 5) #! only buy with 5 or more usdc
        hold = not is_buy
        self.holding_actions.append(hold)
        available_liquidity = self.usdc_held - self.costs_for_action
        # if is_buy and not self.apply_latency():
        if is_buy:
            mapped_buy = 1.0 # know it is buy action -> buy as much as possible
            # mapped_buy = max((action[0] - self.action_factor) / (1 - self.action_factor), 0)
            buy_amount = available_liquidity * mapped_buy
            # buy_amount = self.apply_slippage_to_value(available_liquidity, -self.slippage_level) * mapped_buy
            coins = buy_amount / float(crypto_value)
            self.account_holdings[self.crypto.symbol] += max(coins, 0)
            self.usdc_held = max(self.usdc_held - buy_amount - self.costs_for_action, 0)
            self.bracketSells.append(BracketSellItem(self.crypto, coins, crypto_value))
        # handle bracket selling
        self.handle_bracket_sells(crypto_value)
        self.timestamp += timedelta(minutes=5)
        self.state = self.get_current_state()
        self.past_volumes.append(self.total_volume)
        if len(self.past_volumes) > self.short_term_reward_window:
            self.past_volumes.pop(0)
        self.step_count += 1
        done = False
        info = {}
        return self.state, self.costs_for_action, done, info
    
    def handle_bracket_sells(self, crypto_value: float):
        brackets_to_sell = [i for i, bracket in enumerate(self.bracketSells) if bracket.sellItem(crypto_value)]
        for i in brackets_to_sell:
            sell_amount = self.bracketSells[i].cryptoAmount
            # total_sell_value = sell_amount * crypto_value
            if (self.bracketSells[i].isWinningTrade(crypto_value)):
                # total_sell_value = self.apply_slippage_to_value(sell_amount, -self.slippage_level) * self.bracketSells[i].cryptoValue * self.bracketSells[i].bracketUp #sell for 3% profit
                total_sell_value = sell_amount * self.bracketSells[i].cryptoValue * self.bracketSells[i].bracketUp #sell for 3% profit
            else:
                total_sell_value = sell_amount * self.bracketSells[i].cryptoValue * self.bracketSells[i].bracketDown #sell for 1% loss
                # total_sell_value = self.apply_slippage_to_value(sell_amount, -self.slippage_level) * self.bracketSells[i].cryptoValue * self.bracketSells[i].bracketDown #sell for 1% loss
            self.usdc_held = max(self.usdc_held + total_sell_value - self.get_sell_fees(total_sell_value), 0)
            self.account_holdings[self.crypto.symbol] = 0 # by definition will be all
            # self.account_holdings[self.crypto.symbol] = max(self.account_holdings[self.crypto.symbol] - sell_amount, 0)
            if (total_sell_value > sell_amount * self.bracketSells[i].cryptoValue):
                self.winning_trades += 1
            else:
                self.losing_trades += 1
        self.bracketSells = [bracket for i, bracket in enumerate(self.bracketSells) if i not in brackets_to_sell]

    def get_sell_fees(self, total_value: float):
        return total_value * self.taker_fee * self.transaction_cost_factor

    def get_crypto_value_from_cache(self, symbol: str, timestamp: datetime) -> float:

        timestamp_idx = self.timestamp_to_index.get(timestamp, -1)
        if timestamp_idx == -1:
            raise ValueError(f"Timestamp {timestamp} not in cache")
        close_idx = 0
        return float(self.dataframe_cache_np[timestamp_idx, close_idx])
    
    def get_crypto_values_from_cache_vectorized(self, timestamp: datetime) -> np.ndarray:
        return np.array([self.get_crypto_value_from_cache(self.crypto.symbol, timestamp)], dtype=np.float32)

    def reset_state(self, verbose=True) -> npt.NDArray[np.float16]:
        if verbose:
            if (self.initial_timestamp == self.timestamp and (self.step_count == 0 or self.step_count ==self.total_steps)):
                self.logger.info("Reset at beginning of training...")
                verbose = False
            else:
                self.logger.info(f"Reset: Step: {self.step_count}/{self.total_steps}. ")
                # Log initial state information
                self.logger.info(f"Initial timestamp: {self.initial_timestamp}. Current timestamp: {self.timestamp}")
                self.logger.info(f"USDC: {self.usdc_held}. Crypto: {self.account_holdings}. Total Volume: {self.total_volume}")
                if sum(self.holding_actions) != 0 and len(self.holding_actions) > 0:
                    self.logger.info(f"Resetting training with: {self.total_volume:.2f}$. Holding actions: {sum(self.holding_actions)} / {len(self.holding_actions)}")
        self.step_count = 0
        self.action_factor = 0.2
        self.initial_timestamp = None
        self.maker_fee = 0.0025  # 0.25% based on Advanced 1, might drop
        self.taker_fee = 0.004  # 0.4% based on Advanced 1, might drop
        self.initial_volume = self.initial_volume
        self.total_volume = self.initial_volume
        self.usdc_held = self.initial_volume  # Initialize USDC in memory
        self.account_holdings = {self.crypto.symbol: 0 }
        if verbose:
            self.logger.info(f"after reset: initial volume: {self.initial_volume}. total volume: {self.total_volume}. usdc_held: {self.usdc_held}. account holdings: {self.account_holdings}")
        self.bracketSells:list[BracketSellItem] = []
        self.holding_actions = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.feature_indices = [i for i, feature in enumerate(simulation_columns) if feature in crypto_features]
        
        self.earliest_timestamp = self.get_earliest_timestamp()
        self.maximum_timestamp = self.get_maximum_timestamp()
        self.buffer = 2*self.total_steps
        self.initial_timestamp = self.get_starting_timestamp()
        self.timestamp = self.initial_timestamp
        self.dataframe_cache = self.fetch_all_historical_data_from_file()
        self.dataframe_cache_np = self.convert_dataframe_cache_to_combined_numpy(self.dataframe_cache)

        self.initial_state = self.get_current_state()
        self.past_volumes = []
        self.short_term_reward_window = 10
        return self.initial_state
    
    def map_buy_action_array(self, buy_actions: np.ndarray, action_factor: float) -> np.ndarray:
        return np.maximum(buy_actions - action_factor, 0) / (1 - action_factor)

    def map_sell_action_array(self, sell_actions: np.ndarray, action_factor: float) -> np.ndarray:
        return np.maximum(sell_actions + action_factor - 0.1, action_factor - 1) / (1 - action_factor)
    
    def cost_for_action(self, action: List[float]) -> float:
        #! currently have only 1 action, which can only be buy!
        action_array = np.array(action)
        # buy_mask = action_array >= self.action_factor

        mapped_buy_actions = np.maximum(self.map_buy_action_array(action_array, self.action_factor), 0)
        total_buy_action = 1 if np.sum(mapped_buy_actions) != 0 else 0 #either buy all or not at all
        # total_buy_action = np.sum(mapped_buy_actions)

        # total_buy = mapped_buy_actions[buy_mask].sum()
        # total_buy = total_buy if total_buy != 0 else 1.0
        
        # Calculate transaction volumes
        transaction_volume = self.usdc_held * total_buy_action

        fee_rates = self.taker_fee
        transaction_cost = transaction_volume * fee_rates * self.transaction_cost_factor
        return transaction_cost
        
    
    def get_timestamp_index(self, timestamp: datetime) -> int:
        return self.timestamp_to_index.get(timestamp, -1)
        
    def get_new_crypto_data(self) -> List[float]:
        timestamp_idx = self.get_timestamp_index(self.timestamp)
        if timestamp_idx == -1:
            raise ValueError(f"Timestamp {self.timestamp} not found in index")
        # crypto_data_matrix = self.dataframe_cache_np[timestamp_idx]
        # all_entries = crypto_data_matrix.tolist()
        return self.dataframe_cache_np[timestamp_idx]

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

        transaction_costs = self.costs_for_action
        
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
        
        transaction_costs = self.costs_for_action
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
        
        transaction_costs = self.costs_for_action
        
        # Penalize if holding USDT is less than 0
        holding_penalty = 0
        if action[0] > 0:
            holding_penalty = 5  # Adjust the penalty factor as needed

        reward = net_return * 0.25 + cumulative_reward * 1.0 - transaction_costs * 0.5 - holding_penalty
        reward *= 5
        if reward > 0:
            reward *= asymmetry_factor
        return reward
