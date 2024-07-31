from typing import Any, Dict, List, Tuple
import tqdm
import pandas as pd
from coinbase_api.ml_models.data_handlers.abstract_data_handler import AbstractDataHandler
from coinbase_api.utilities.utils import calculate_total_volume, initialize_default_cryptos
import numpy as np
import numpy.typing as npt
from coinbase_api.models.models import AbstractOHLCV, Account, Prediction, CryptoMetadata
from coinbase_api.constants import crypto_models, crypto_features, crypto_predicted_features, crypto_extra_features
from coinbase_api.enums import Actions, Database
from datetime import datetime, timedelta
from django.db import transaction
import random
import concurrent.futures
from django.db import transaction, connection
import time

class SimulationDataHandler(AbstractDataHandler):
    def __init__(self, initial_volume=1000, total_steps=1024, transaction_cost_factor=1.0) -> None:
        self.initial_timestamp = None
        self.lstm_sequence_length = 100
        self.total_steps = total_steps
        self.transaction_cost_factor = transaction_cost_factor
        self.action_factor = 0.5
        self.step_count = 0
        self.usdc_held = 0
        self.minimum_number_of_cryptos = 100
        self.maker_fee = 0.004  # 0.4%
        self.taker_fee = 0.006  # 0.6%
        self.initial_volume: float = initial_volume
        self.database: Database = Database.SIMULATION.value
        self.crypto_models: List[AbstractOHLCV] = crypto_models
        self.total_volume = initial_volume
        self.account_holdings = [0 for _ in self.crypto_models]
        self.earliest_timestamp = self.get_earliest_timestamp()
        self.maximum_timestamp = self.get_maximum_timestamp()
        self.buffer = 1024
        self.initial_timestamp = self.get_starting_timestamp()
        self.timestamp = self.initial_timestamp
        self.initial_prices = self.get_initial_crypto_prices()
        self.dataframe_cache = self.fetch_all_historical_data()
        self.prediction_cache = self.fetch_all_prediction_data()
        self.prepare_simulation_database()
        self.state = self.get_current_state()
        self.past_volumes = []
        self.short_term_reward_window = 10

    def fetch_all_historical_data(self) -> dict[str, pd.DataFrame]:
        dataframes = {}
        start_timestamp = self.timestamp - timedelta(hours=self.total_steps + self.lstm_sequence_length - 1)
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
            dataframes[crypto_model.symbol] = df

        return dataframes

    def fetch_all_prediction_data(self) -> dict[str, dict[datetime, List[float]]]:
        start_time = time.time()
        prediction_data = {}
        start_timestamp = self.timestamp - timedelta(hours=self.total_steps + self.lstm_sequence_length - 1)
        end_timestamp = self.timestamp + timedelta(hours=self.total_steps)

        for crypto_model in self.crypto_models:
            predictions = Prediction.objects.using(Database.HISTORICAL.value).filter(
                timestamp_predicted_for__range=(start_timestamp, end_timestamp),
                crypto=crypto_model.symbol,
                model_name__in=['LSTM', 'XGBoost'],
                predicted_field__in=['close_higher_shifted_1h', 'close_higher_shifted_24h', 'close_higher_shifted_168h']
            ).values(
                'timestamp_predicted_for', 'model_name', 'predicted_field', 'predicted_value'
            )

            crypto_predictions = {}
            for pred in predictions:
                timestamp = pred['timestamp_predicted_for']
                if timestamp not in crypto_predictions:
                    crypto_predictions[timestamp] = [0.0] * 6  # 3 for LSTM and 3 for XGBoost
                index = 0 if pred['predicted_field'] == 'close_higher_shifted_1h' else 1 if pred['predicted_field'] == 'close_higher_shifted_24h' else 2
                index += 0 if pred['model_name'] == 'LSTM' else 3
                crypto_predictions[timestamp][index] = pred['predicted_value']

            prediction_data[crypto_model.symbol] = crypto_predictions
        end_time = time.time()
        print(f'time to fetch all predictions: {end_time - start_time} = {len(crypto_models)} * {(end_time - start_time) / len(crypto_models)}')
        return prediction_data

    def get_initial_crypto_prices(self) -> Dict[str, float]:
        values = {}
        for crypto_model in self.crypto_models:
            try:
                values[crypto_model.symbol] = crypto_model.objects.using(Database.HISTORICAL.value).filter(timestamp=self.timestamp).get().close
            except crypto_model.DoesNotExist:
                values[crypto_model.symbol] = crypto_model.default_entry(timestamp=datetime(year=2020, month=1, day=1)).close
            except AttributeError:
                values[crypto_model.symbol] = crypto_model.default_entry(timestamp=datetime(year=2020, month=1, day=1)).close
            except crypto_model.MultipleObjectsReturned:
                values[crypto_model.symbol] = crypto_model.default_entry(timestamp=datetime(year=2020, month=1, day=1)).close
        return values

    def get_reward_ratios_for_current_timestep(self, amount=5) -> Dict[str, float]:
        ratios = {}
        for crypto_model in self.crypto_models:
            value = self.get_latest_value(crypto_model)
            initial_price = self.initial_prices.get(crypto_model.symbol, 0)
            ratios[crypto_model.symbol] = value / initial_price if initial_price != 0 else 0

        sorted_ratios = dict(sorted(ratios.items(), key=lambda item: -item[1]))
        return {k: sorted_ratios[k] for k in list(sorted_ratios)[:amount]}

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
        # start_time = time.time()
        
        self.total_volume, self.account_holdings, self.usdc_held = self.get_account_and_volume_info()
        # get_account_and_volume_info_time = time.time() - start_time

        # start_time = time.time()
        self.new_crypto_data = self.get_new_crypto_data()
        # get_new_crypto_data_time = time.time() - start_time

        # print(f"get_account_and_volume_info_time: {get_account_and_volume_info_time:.4f}s")
        # print(f"get_new_crypto_data_time: {get_new_crypto_data_time:.4f}s")

        return np.array([self.total_volume, self.usdc_held] + self.account_holdings + self.new_crypto_data)


    def get_account_and_volume_info(self) -> Tuple[float, List[float], float]:
        total_volume = 0.0
        account_holdings = []
        usdc_held = 0.0
        
        accounts = Account.objects.using(self.database).filter(
            name__in=[f'{crypto.symbol} Wallet' for crypto in self.crypto_models] + ['USDC Wallet']
        ).select_related()
        
        # crypto_prices = {
        #     crypto.symbol: crypto.objects.using(self.database).latest('timestamp').close for crypto in self.crypto_models
        # }
        
        for account in accounts:
            if account.name == 'USDC Wallet':
                usdc_held = account.value
                total_volume += usdc_held
            else:
                symbol = account.name.split()[0]
                # price = crypto_prices.get(symbol, 0)
                price = self.get_crypto_value_from_cache(symbol, self.timestamp)
                total_volume += account.value * price
                account_holdings.append(account.value)
        
        return total_volume, account_holdings, usdc_held

    def map_buy_action(self, buy_action: float, action_factor: float) -> float:
        return (buy_action - action_factor) / (1 - action_factor)

    def map_sell_action(self, sell_action: float, action_factor: float) -> float:
        return (max(sell_action + action_factor - 0.1, action_factor - 1)) / (1 - action_factor)
    
    def get_dataframes_subset(self) -> dict[str, pd.DataFrame]:
        subset = {}
        for symbol, df in self.dataframe_cache.items():
            subset[symbol] = df.loc[self.timestamp - timedelta(hours=self.lstm_sequence_length - 1):self.timestamp]
        return subset

    def update_state(self, action) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        start_time = time.time()
        # interval_times = []

        # interval_start = time.time()
        costs_for_action = self.cost_for_action(action)
        # interval_times.append(("Cost Calculation", time.time() - interval_start))

        # interval_start = time.time()
        buy_indices = [(idx, i) for idx, i in enumerate(action) if i > self.action_factor and self.crypto_models[idx].symbol != 'USDT']
        sell_indices = [(idx, i) for idx, i in enumerate(action) if i < -self.action_factor]
        usdt_action = action[self.crypto_models.index(next(cm for cm in self.crypto_models if cm.symbol == 'USDT'))]
        hold_usdc = usdt_action < 0
        # interval_times.append(("Get Indices", time.time() - interval_start))

        # interval_start = time.time()
        usdc_account = self.get_crypto_account('USDC')
        available_liquidity = self.get_liquidity() - sum(costs_for_action)
        # interval_times.append(("Get USDC Account and Liquidity", time.time() - interval_start))

        # interval_start = time.time()
        N = 5
        top_buy_indices = sorted(buy_indices, key=lambda x: -x[1])[:N]
        total_buy_actions = sum([self.map_buy_action(buy_action, self.action_factor) for idx, buy_action in top_buy_indices])
        individual_liquidity = available_liquidity / total_buy_actions if total_buy_actions > 0 else 0
        # interval_times.append(("Select Top Cryptos for Buying", time.time() - interval_start))

        # interval_start = time.time()
        accounts = {acc.name: acc for acc in Account.objects.using(self.database).filter(
            name__in=[f'{crypto.symbol} Wallet' for crypto in self.crypto_models] + ['USDC Wallet']
        ).select_related()}
        
        crypto_updates: List[Account] = []
        usdc_updates: List[Account] = []
        if not hold_usdc:
            for idx, buy_action in top_buy_indices:
                crypto_model = self.crypto_models[idx]
                crypto_account = accounts.get(f'{crypto_model.symbol} Wallet', Account(name=f'{crypto_model.symbol} Wallet', value=0))
                cost_for_action = costs_for_action[idx]
                buy_action_mapped = self.map_buy_action(buy_action, self.action_factor)
                crypto_value = self.get_crypto_value_from_cache(crypto_model.symbol, self.timestamp)
                buy_amount = individual_liquidity * buy_action_mapped if crypto_value else 0
                crypto_account.value += (buy_amount - cost_for_action) / crypto_value if crypto_value else 0
                usdc_account.value -= buy_amount
                crypto_updates.append(crypto_account)
                usdc_updates.append(usdc_account)
        # interval_times.append(("Perform Buy Actions", time.time() - interval_start))

        # interval_start = time.time()

        for idx, sell_action in sell_indices:
            crypto_model = self.crypto_models[idx]
            crypto_account = accounts.get(f'{crypto_model.symbol} Wallet', Account(name=f'{crypto_model.symbol} Wallet', value=0))
            cost_for_action = costs_for_action[idx]
            sell_action_mapped = self.map_sell_action(sell_action, self.action_factor)
            crypto_value = self.get_crypto_value_from_cache(crypto_model.symbol, self.timestamp)
            sell_amount = abs(crypto_account.value * sell_action_mapped) if crypto_value else 0
            usdc_account.value += sell_amount * crypto_value - cost_for_action if crypto_value else 0
            crypto_account.value -= sell_amount
            crypto_updates.append(crypto_account)
            usdc_updates.append(usdc_account)
        # interval_times.append(("Perform Sell Actions", time.time() - interval_start))

        # interval_start = time.time()

        with transaction.atomic(using=self.database):
            Account.objects.using(self.database).bulk_update(crypto_updates, ['value'])
            Account.objects.using(self.database).bulk_update(usdc_updates, ['value'])
        # interval_times.append(("Bulk Save Updates", time.time() - interval_start))

        # interval_start = time.time()

        new_timestamp = self.timestamp + timedelta(hours=1)
        done = False
        # interval_times.append(("Fetch New Data", time.time() - interval_start))

        # interval_start = time.time()

        self.timestamp = new_timestamp
        self.state = self.get_current_state()
        self.past_volumes.append(self.total_volume)
        if len(self.past_volumes) > self.short_term_reward_window:
            self.past_volumes.pop(0)
        # interval_times.append(("Get Current State", time.time() - interval_start))
        end_time = time.time()
        total_time = end_time - start_time

        print(f'Total time for update_state: {total_time:.2f} seconds')

        # for interval_name, interval_duration in interval_times:
        #     print(f'{interval_name}: {interval_duration:.2f} seconds ({(interval_duration / total_time) * 100:.2f}%)')

        self.step_count += 1
        info = {}
        return self.state, sum(costs_for_action), done, info

    def get_crypto_value_from_cache(self, symbol: str, timestamp: datetime) -> float:
        df = self.dataframe_cache[symbol]
        if timestamp in df.index:
            try:
                return df.at[timestamp, 'close']
            except Exception as e:
                return 0.0
        return 0.0

    def get_crypto_data_from_cache(self, symbol: str, timestamp: datetime) -> AbstractOHLCV:
        df = self.dataframe_cache[symbol]
        crypto_model = [crypto for crypto in crypto_models if crypto.symbol == symbol].pop(0)
        if timestamp in df.index:
            row = df.loc[timestamp]
            return crypto_model(
                timestamp=row.name,
                volume=row['volume'],
                sma=row['sma'],
                ema=row['ema'],
                rsi=row['rsi'],
                macd=row['macd'],
                bollinger_high=row['bollinger_high'],
                bollinger_low=row['bollinger_low'],
                vmap=row['vmap'],
                percentage_returns=row['percentage_returns'],
                log_returns=row['log_returns']
            )
        return crypto_model.default_entry(timestamp=timestamp)

    def get_current_state_output(self, action) -> str:
        reward_q = self.get_reward(action)
        reward_ratios = self.get_reward_ratios_for_current_timestep()
        reward_string = f'gain: {self.total_volume / 1000:.3f},'
        # Fetch all account information in a single query
        accounts = {acc.name: acc for acc in Account.objects.using(self.database).filter(
            name__in=[f'{crypto.symbol} Wallet' for crypto in self.crypto_models] + ['USDC Wallet']
        ).select_related()}

        # Fetch crypto prices from cache
        crypto_prices = {
            crypto.symbol: self.get_crypto_value_from_cache(crypto.symbol, self.timestamp)
            for crypto in self.crypto_models
        }
        for key, value in reward_ratios.items():
            crypto_model = self.get_crypto_model_by_symbol(key)
            account_value, price = self.get_crypto_account_value(crypto_model, accounts, crypto_prices)
            reward_string += self.get_output_for_crypto(crypto_model, value, account_value, price)

        cost = sum(self.cost_for_action(action))
        liquidity_string = self.get_liquidity_string(self.get_reward_ratios_for_current_timestep(len(self.crypto_models)))

        return f"\n\n\n{self.get_step_count()}/{self.get_total_steps()}: time: {self.timestamp}\n\tcurrent volume: {self.total_volume:.2f}\n\tusdc: {self.usdc_held:.2f}\n\treward {reward_q:.2f}\n\tcost: {cost:.2f}\n\t{reward_string}\n\tcryptos: {liquidity_string}"

    def get_liquidity_string(self, reward_ratios: Dict[str, float]) -> str:
        # Fetch all account information in a single query
        accounts = {acc.name: acc for acc in Account.objects.using(self.database).filter(name__in=[f'{crypto.symbol} Wallet' for crypto in self.crypto_models] + ['USDC Wallet']).select_related()}

        # Fetch crypto prices from cache
        crypto_prices = {
            crypto.symbol: self.get_crypto_value_from_cache(crypto.symbol, self.timestamp)
            for crypto in self.crypto_models
        }

        # Create a dictionary with account values and prices
        crypto_model_values = {
            crypto_model.symbol: (accounts.get(f'{crypto_model.symbol} Wallet', Account(name=f'{crypto_model.symbol} Wallet', value=0)).value, crypto_prices[crypto_model.symbol])
            for crypto_model in self.crypto_models
        }

        # Sort the dictionary based on the product of account value and price
        sorted_crypto_values = dict(sorted(crypto_model_values.items(), key=lambda item: -item[1][0] * item[1][1]))

        # Generate the output string
        ret_string = ''
        for idx, (symbol, (account_value, price)) in enumerate(sorted_crypto_values.items()):
            crypto_model = self.get_crypto_model_by_symbol(symbol)
            reward_ratio = reward_ratios.get(symbol, 0)
            ret_string += self.get_output_for_crypto(crypto_model, reward_ratio, account_value, price)
            if idx > 5:
                break

        return ret_string

    def get_latest_value(self, crypto_model: AbstractOHLCV) -> float:
        try:
            return self.get_crypto_value_from_cache(crypto_model.symbol, self.timestamp)
        except (crypto_model.DoesNotExist, AttributeError):
            return crypto_model.default_entry(timestamp=datetime(year=2020, month=1, day=1)).close

    def get_crypto_model_by_symbol(self, symbol: str):
        return next(crypto for crypto in self.crypto_models if crypto.symbol == symbol)

    def get_crypto_account_value(self, crypto_model: AbstractOHLCV, accounts: Dict[str, Account], crypto_prices: Dict[str, float]) -> Tuple[float, float]:
        account_value = accounts.get(f'{crypto_model.symbol} Wallet', Account(name=f'{crypto_model.symbol} Wallet', value=0)).value
        price = crypto_prices.get(crypto_model.symbol, 0)
        return account_value, price

    def volume_decay(self, factor: float = 0.002) -> None:
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
        self.action_factor = 0.2
        self.initial_timestamp = None
        self.maker_fee = 0.004  # 0.4%
        self.taker_fee = 0.006  # 0.6%
        self.initial_volume = self.initial_volume
        self.database = Database.SIMULATION.value
        self.crypto_models: List[AbstractOHLCV] = crypto_models
        self.total_volume = self.initial_volume
        self.account_holdings = [0 for _ in self.crypto_models]
        self.earliest_timestamp = self.get_earliest_timestamp()
        self.maximum_timestamp = self.get_maximum_timestamp()
        self.buffer = 1024
        self.initial_timestamp = self.get_starting_timestamp()
        self.timestamp = self.initial_timestamp
        self.initial_prices = self.get_initial_crypto_prices()
        self.dataframe_cache = self.fetch_all_historical_data()
        self.prediction_cache = self.fetch_all_prediction_data()
        self.prepare_simulation_database()
        self.initial_state = self.get_current_state()
        return self.initial_state

    def cost_for_action(self, action: List[float]) -> List[float]:
        all_costs = []
        total_buy_action = sum(max(self.map_buy_action(crypto_action, self.action_factor), 0) for crypto_action in action if crypto_action >= self.action_factor)
        if total_buy_action == 0:
            total_buy_action = 1

        # Fetch all account information in a single query
        accounts = {acc.name: acc for acc in Account.objects.using(self.database).filter(name__in=[f'{crypto.symbol} Wallet' for crypto in self.crypto_models] + ['USDC Wallet']).select_related()}
        available_liquidity = accounts.get('USDC Wallet', Account(name='USDC Wallet', value=0)).value

        for idx, crypto_action in enumerate(action):
            if -self.action_factor < crypto_action < self.action_factor:
                all_costs.append(0)
                continue
            is_buy = crypto_action >= self.action_factor
            crypto = self.crypto_models[idx]
            try:
                transaction_volume = self.calculate_transaction_volume(crypto, is_buy, crypto_action, total_buy_action, available_liquidity, accounts)
            except Account.DoesNotExist:
                continue
            fee_rate = self.maker_fee if is_buy else self.taker_fee
            transaction_cost = transaction_volume * fee_rate * self.transaction_cost_factor
            all_costs.append(transaction_cost)
        return all_costs


    def calculate_transaction_volume(self, crypto: AbstractOHLCV, is_buy: bool, factor: float, total_buy_action: float, available_liquidity: float, accounts: Dict[str, Account]) -> float:
        if is_buy:
            price = self.get_crypto_value_from_cache(crypto.symbol, self.timestamp)
            proportion = self.map_buy_action(factor, self.action_factor) / total_buy_action
            if price == 0:
                transaction_volume = 0
            else:
                transaction_volume = available_liquidity * proportion
        else:
            crypto_account = accounts.get(f'{crypto.symbol} Wallet', Account(name=f'{crypto.symbol} Wallet', value=0))
            price = self.get_crypto_value_from_cache(crypto.symbol, self.timestamp)
            transaction_volume = price * crypto_account.value * abs(factor)
        return transaction_volume


    def get_crypto_features(self) -> List[str]:
        return crypto_features

    def get_crypto_predicted_features(self) -> List[str]:
        return crypto_predicted_features

    def crypto_to_list(self, crypto: AbstractOHLCV) -> List[float]:
        return [getattr(crypto, fieldname) for fieldname in self.get_crypto_features()]

    def get_new_prediction_data(self, timestamp: datetime) -> Dict[str, List[float]]:
        all_entries = {crypto.symbol: [0.0] * 6 for crypto in self.crypto_models}
        
        for crypto in self.crypto_models:
            if timestamp in self.prediction_cache[crypto.symbol]:
                all_entries[crypto.symbol] = self.prediction_cache[crypto.symbol][timestamp]
        
        return all_entries

    def prepare_simulation_database(self) -> None:
        print(f'Preparing simulation.')

        def process_crypto_model(crypto_model: AbstractOHLCV):
            crypto_model.objects.using(self.database).all().delete()

            historical_data = crypto_model.objects.using(Database.HISTORICAL.value).filter(
                timestamp__lte=self.timestamp,
                timestamp__gte=self.timestamp - timedelta(days=30)
            )

            new_instances = []
            for obj in historical_data:
                new_instance = self.get_new_instance(crypto_model, obj)
                new_instances.append(new_instance)

            if len(new_instances) <= 30 * 24:
                for _ in range(30 * 24 - len(new_instances)):
                    new_instances.insert(0, crypto_model.default_entry(self.timestamp - timedelta(hours=len(new_instances))))

            with transaction.atomic(using=self.database):
                crypto_model.objects.using(self.database).bulk_create(new_instances)

        l = len(self.crypto_models)
        print(f'restoring database for a total of {l} cryptos...')
        with tqdm.tqdm(total=l) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
                futures = [executor.submit(process_crypto_model, crypto_model) for crypto_model in self.crypto_models]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f'Error processing crypto model: {e}')
                    finally:
                        pbar.update(1)
                        connection.close()
        self.reset_account_data()

    def reset_account_data(self) -> None:
        account_data = Account.objects.using(self.database).all()
        account_data.delete()
        initialize_default_cryptos(initial_volume=self.initial_volume, database=self.database)

    def get_new_instance(self, crypto_model: AbstractOHLCV, instance: AbstractOHLCV) -> AbstractOHLCV:
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

    def get_new_crypto_data(self) -> List[float]:
        all_entries = []
        prediction_data = self.get_new_prediction_data(self.timestamp)
        
        for crypto in self.crypto_models:
            crypto_latest = self.get_crypto_data_from_cache(crypto.symbol, self.timestamp)
            all_entries.extend(self.crypto_to_list(crypto_latest))
            all_entries.extend(prediction_data[crypto.symbol])
        
        return all_entries

    def get_step_count(self) -> int:
        return self.step_count

    def get_total_steps(self) -> int:
        return self.total_steps

    def get_output_for_crypto(self, crypto_model: AbstractOHLCV, reward_ratio, account_value=None, price=None) -> str:
        return f'\n\t\t{crypto_model.symbol}:\t{account_value:.2f} =\t{account_value * price:.2f} (reward ratio: {reward_ratio:.2f})'

    def get_reward(self, action: Actions) -> float:
        net_return = (self.total_volume / self.initial_volume) - 1
        reward = net_return * 0.1  # Scale the reward as needed
        asymmetry_factor = 2.0  # Adjust this factor to control asymmetry

        if reward > 0:
            reward *= asymmetry_factor

        if len(self.past_volumes) >= self.short_term_reward_window:
            past_volume = np.sum(self.past_volumes[-self.short_term_reward_window:]) / self.short_term_reward_window
            short_term_gain = (self.total_volume - past_volume) / past_volume
            reward += short_term_gain * 5
        reward *= 5

        return reward
