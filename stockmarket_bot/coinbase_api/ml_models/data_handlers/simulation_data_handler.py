from typing import Any, Dict, List, Tuple
import tqdm
from coinbase_api.ml_models.data_handlers.abstract_data_handler import AbstractDataHandler
from coinbase_api.utilities.utils import calculate_total_volume, initialize_default_cryptos
import numpy as np
import numpy.typing as npt
from coinbase_api.models.models import AbstractOHLCV, Account, Prediction, CryptoMetadata
from coinbase_api.constants import crypto_models, crypto_features, crypto_predicted_features, crypto_extra_features
from coinbase_api.utilities.prediction_handler import PredictionHandler
from coinbase_api.enums import Actions, Database
from datetime import datetime, timedelta
from django.db import transaction
import random
import concurrent.futures
from django.db import transaction, connection

class SimulationDataHandler(AbstractDataHandler):
    def __init__(self, initial_volume = 1000, total_steps = 1024, transaction_cost_factor=1.0) -> None:
        self.initial_timestamp = None
        self.total_steps = total_steps
        self.transaction_cost_factor = transaction_cost_factor
        self.action_factor = 0.5
        self.step_count = 0
        self.usdc_held = 0
        self.minimum_number_of_cryptos = 100
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
        self.prepare_simulation_database()
        self.prepare_predictions()
        self.state = self.get_current_state()
        self.past_volumes = []
        self.short_term_reward_window = 10

    def prepare_predictions(self):
        # refactored from prepare_simulation_data_handling for consistency issues
        self.prediction_handler = PredictionHandler(lstm_sequence_length=100, database=self.database, timestamp=self.timestamp)
        # 1. Delete all predictions and crypto model entries
        print(f'Deleting predictions')
        self.prediction_handler.restore_prediction_database()
        # 4. Get initial predictions in
        self.prediction_handler.predict()

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
                all_timestamps.append(val if val != None else (datetime(year=2024, month=6, day=20),))
            except crypto_model.DoesNotExist:
                print(f'did not exist: {crypto_model.symbol}')
                all_timestamps.append(crypto_model.default_entry(timestamp=datetime(year=2024, month=6, day=20))) # TODO Need to change this manually. However only relevant, if there is no data at all
            except AttributeError:
                print(f'attribute error: {crypto_model.symbol}')
                all_timestamps.append(crypto_model.default_entry(timestamp=datetime(year=2024, month=6, day=20))) # TODO Need to change this manually. However only relevant, if there is no data at all
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
        self.total_volume = calculate_total_volume(database=self.database)
        self.account_holdings = self.get_account_holdings()
        self.new_crypto_data = self.get_new_crypto_data()
        self.usdc_held = self.get_liquidity()
        return np.array([self.total_volume, self.usdc_held] + self.account_holdings + self.new_crypto_data)
        
    def map_buy_action(self, buy_action: float, action_factor: float)-> float:
        return (buy_action - action_factor) / (1 - action_factor)
    
    def map_sell_action(self, sell_action: float, action_factor: float)-> float:
        return (sell_action + action_factor) / (1 - action_factor)
    
    def update_state(self, action) -> Tuple[npt.NDArray[np.float16], float, bool, Dict[Any, Any]]:
        start_time = datetime.now()
        costs_for_action = self.cost_for_action(action) # 0.4s
        buy_indices = [(idx, i) for idx, i in enumerate(action) if i > self.action_factor and self.crypto_models[idx].symbol != 'USDT']
        sell_indices = [(idx, i) for idx, i in enumerate(action) if i < -self.action_factor]

        usdt_action = action[self.crypto_models.index(next(cm for cm in self.crypto_models if cm.symbol == 'USDT'))]
        hold_usdc = usdt_action < 0
        
        # Perform buy actions on DB
        usdc_account = self.get_crypto_account('USDC')
        available_liquidity = self.get_liquidity() - sum(costs_for_action)
        
        # Select top N cryptocurrencies for buying
        N = 5  # You can adjust this number based on your preference
        top_buy_indices = sorted(buy_indices, key=lambda x: -x[1])[:N]
        
        # Distribute liquidity among the top N cryptocurrencies
        total_buy_actions = sum([self.map_buy_action(buy_action, self.action_factor) for idx, buy_action in top_buy_indices])
        individual_liquidity = available_liquidity / total_buy_actions if total_buy_actions > 0 else 0
    
        crypto_updates:List[Account] = []
        usdc_updates:List[Account] = []
        if (not hold_usdc): # only buy when allowed
            for idx, buy_action in top_buy_indices:
                crypto_model = self.crypto_models[idx]
                crypto_account = self.get_crypto_account(crypto_model.symbol)
                cost_for_action = costs_for_action[idx]
                buy_action_mapped = self.map_buy_action(buy_action, self.action_factor)
                crypto_value = crypto_model.objects.using(self.database).latest('timestamp').close
                buy_amount = individual_liquidity * buy_action_mapped if crypto_value else 0
                crypto_account.value += (buy_amount - cost_for_action) / crypto_value if crypto_value else 0
                usdc_account.value -= buy_amount
                crypto_updates.append(crypto_account)
                usdc_updates.append(usdc_account)

        for idx, sell_action in sell_indices:
            crypto_model = self.crypto_models[idx]
            crypto_account = self.get_crypto_account(crypto_model.symbol)
            cost_for_action = costs_for_action[idx]
            sell_action_mapped = self.map_sell_action(sell_action, self.action_factor)
            crypto_value = crypto_model.objects.using(self.database).latest('timestamp').close
            sell_amount = abs(crypto_account.value * sell_action_mapped) if crypto_value else 0
            usdc_account.value += sell_amount * crypto_value - cost_for_action if crypto_value else 0
            crypto_account.value -= sell_amount
            crypto_updates.append(crypto_account)
            usdc_updates.append(usdc_account)

        # Bulk save updates
        with transaction.atomic(using=self.database):
            Account.objects.using(self.database).bulk_update(crypto_updates, ['value'])
            Account.objects.using(self.database).bulk_update(usdc_updates, ['value'])
        new_timestamp = self.timestamp + timedelta(hours=1)
        done = False
        
        new_data_instances = []
        for crypto in self.crypto_models:
            historical_data = crypto.objects.using(Database.HISTORICAL.value).filter(timestamp=new_timestamp).first()
            if historical_data is None:
                new_data = crypto.default_entry(timestamp=new_timestamp)
            else:
                new_data = self.get_new_instance(crypto_model=crypto, instance=historical_data)
            new_data_instances.append([new_data])

        # Bulk create new data instances
        with transaction.atomic(using=self.database):
            for crypto_model, instances in zip(self.crypto_models, new_data_instances):
                crypto_model.objects.using(self.database).bulk_create(instances)
        self.prediction_handler.timestamp = new_data.timestamp
        self.timestamp = new_timestamp
        self.prediction_handler.predict() # 33s
        self.state = self.get_current_state() # 2s
        self.past_volumes.append(self.total_volume)
        if len(self.past_volumes) > self.short_term_reward_window:
            self.past_volumes.pop(0)
        print(f'{self.step_count}/{self.total_steps}\tend time:\t{datetime.now() - start_time}')
        self.step_count += 1
        info = {}
        return self.state, sum(costs_for_action), done, info

    def get_current_state_output(self, action) -> str:
        reward_q = self.get_reward(action)
        reward_ratios = self.get_reward_ratios_for_current_timestep()
        reward_string = f'gain: {self.total_volume / 1000:.3f},'
        
        for key, value in reward_ratios.items():
            crypto_model = self.get_crypto_model_by_symbol(key)
            reward_string += self.get_output_for_crypto(crypto_model, value)
        
        cost = sum(self.cost_for_action(action))
        liquidity_string = self.get_liquidity_string(self.get_reward_ratios_for_current_timestep(len(self.crypto_models)))
        
        return f"\n\n\n{self.get_step_count()}/{self.get_total_steps()}: time: {self.timestamp}\n\tcurrent volume: {self.total_volume:.2f}\n\tusdc: {self.usdc_held:.2f}\n\treward {reward_q:.2f}\n\tcost: {cost:.2f}\n\t{reward_string}\n\tcryptos: {liquidity_string}"

    def get_liquidity_string(self, reward_ratios: Dict[str, float]) -> str:
        crypto_model_values = {
            crypto_model.symbol: self.get_crypto_account_value(crypto_model)
            for crypto_model in self.crypto_models
        }

        sorted_crypto_values = dict(sorted(crypto_model_values.items(), key=lambda item: -item[1][0] * item[1][1]))

        ret_string = ''
        for idx, (symbol, (account_value, price)) in enumerate(sorted_crypto_values.items()):
            crypto_model = self.get_crypto_model_by_symbol(symbol)
            # TODO: Why does this sometimes return 0, i.e. on the cryptos with most value part??
            reward_ratio = reward_ratios.get(symbol, 0)
            ret_string += self.get_output_for_crypto(crypto_model, reward_ratio, account_value, price)
            if idx > 5:
                break
    
        return ret_string
    
    def get_latest_value(self, crypto_model: AbstractOHLCV) -> float:
        try:
            return crypto_model.objects.using(Database.HISTORICAL.value).filter(timestamp=self.timestamp).first().close
        except (crypto_model.DoesNotExist, AttributeError):
            return crypto_model.default_entry(timestamp=datetime(year=2020, month=1, day=1)).close
    
    def get_crypto_model_by_symbol(self, symbol: str):
        return next(crypto for crypto in self.crypto_models if crypto.symbol == symbol)
    
    def get_crypto_account_value(self, crypto_model: AbstractOHLCV) -> Tuple[float, float]:
        try:
            account_value = self.get_crypto_account(crypto_model.symbol).value
        except Account.DoesNotExist:
            account_value = 0
        try:
            price = crypto_model.objects.using(self.database).latest("timestamp").close
        except (crypto_model.DoesNotExist, AttributeError):
            price = 0
        return account_value, price
    
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
        self.action_factor = 0.2
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
        self.prepare_simulation_database()
        self.prepare_predictions()
        # self.prediction_handler = PredictionHandler(lstm_sequence_length=100, database=self.database, timestamp=self.timestamp)
        self.initial_state = self.get_current_state()
        return self.initial_state
    
    def cost_for_action(self, action: List[float]) -> float:
        all_costs = []
        total_buy_action = sum(max(self.map_buy_action(crypto_action, self.action_factor), 0) for crypto_action in action if crypto_action >= self.action_factor)
        # If no buy actions, skip the distribution calculation
        if total_buy_action == 0:
            total_buy_action = 1
        for idx, crypto_action in enumerate(action):
            if -self.action_factor < crypto_action < self.action_factor:
                all_costs.append(0)
                continue
            is_buy = crypto_action >= self.action_factor
            crypto = self.crypto_models[idx]
            try:
                transaction_volume = self.calculate_transaction_volume(crypto, is_buy, crypto_action, total_buy_action)
            except Account.DoesNotExist:
                continue
            fee_rate = self.maker_fee if is_buy else self.taker_fee  # Assuming maker fee for buy, taker fee for sell
            transaction_cost = transaction_volume * fee_rate * self.transaction_cost_factor
            all_costs.append(transaction_cost)
        return all_costs
    
    def calculate_transaction_volume(self, crypto: AbstractOHLCV, is_buy: bool, factor: float, total_buy_action: float) -> float:
        if is_buy:
            crypto_account = self.get_crypto_account('USDC')
            available_liquidity = crypto_account.value
            price = crypto.objects.using(self.database).latest('timestamp').close
            proportion = self.map_buy_action(factor, self.action_factor) / total_buy_action
            if (price == 0):
                transaction_volume = 0
            else:
                transaction_volume = available_liquidity * proportion
        else:
            crypto_account = self.get_crypto_account(crypto.symbol)
            price = crypto.objects.using(self.database).latest('timestamp').close
            transaction_volume = price * crypto_account.value * abs(factor)
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
                    all_entries.append(0.0)
        return all_entries
    
    def prepare_simulation_database(self) -> None:
        print(f'Preparing simulation.')

        # Function to delete and recreate data for a single crypto model
        def process_crypto_model(crypto_model: AbstractOHLCV):
            # print(f'Deleting model data for {crypto_model.symbol}')
            crypto_model.objects.using(self.database).all().delete()

            # print(f'Fetching model data for {crypto_model.symbol}')
            historical_data = crypto_model.objects.using(Database.HISTORICAL.value).filter(
                timestamp__lte=self.timestamp,
                timestamp__gte=self.timestamp - timedelta(days=30)
            )

            new_instances = []
            for obj in historical_data:
                new_instance = self.get_new_instance(crypto_model, obj)
                new_instances.append(new_instance)

            # Prepend default entries if less than 30 days of data found
            if len(new_instances) <= 30*24:
                for _ in range(30*24 - len(new_instances)):
                    new_instances.insert(0, crypto_model.default_entry(self.timestamp - timedelta(hours=len(new_instances))))

            with transaction.atomic(using=self.database):
                crypto_model.objects.using(self.database).bulk_create(new_instances)

        # Use ThreadPoolExecutor to parallelize processing of crypto models
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
        # 3. Prepare the account database
        self.reset_account_data()
        


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
                account_holdings.append(0.0)
                continue
            account_holdings.append(account.value)
        return account_holdings
    
    def get_new_crypto_data(self) -> List[float]:
        all_entries = []
        for crypto in self.crypto_models:
            crypto_latest = crypto.objects.using(self.database).latest('timestamp')
            if (crypto_latest == None):
                crypto_latest = crypto.default_entry(self.timestamp)
            all_entries = all_entries + self.crypto_to_list(crypto_latest)
            all_entries = all_entries + self.get_new_prediction_data(crypto, crypto_latest.timestamp)
        return all_entries

    def get_step_count(self) -> int:
        return self.step_count
    
    def get_total_steps(self) -> int:
        return self.total_steps
    
    # def get_current_state_output(self, action) -> str:
    #     reward_q = self.get_reward(action)
    #     reward_ratios = self.get_reward_ratios_for_current_timestep()
    #     reward_string = f'gain: {self.total_volume/1000:.3f},'
    #     for key in reward_ratios:
    #         crypto_model = [crypto for crypto in crypto_models if crypto.symbol == key][0]
    #         reward_string = reward_string + self.get_output_for_crypto(reward_ratios=reward_ratios, crypto_model=crypto_model)
    #     return f"\n\n\n{self.get_step_count()}/{self.get_total_steps()}: time: {self.timestamp}\n\tcurrent volume: {self.total_volume:.2f}\n\tusdc: {self.usdc_held:.2f}\n\treward {reward_q:.2f}\n\tcost: {sum(self.cost_for_action(action)):.2f}\n\t{reward_string}\n\tcryptos: {self.get_liquidity_string()}"
    
        
    def get_output_for_crypto(self, crypto_model: AbstractOHLCV, reward_ratio, account_value=None, price=None) -> str:
        if account_value is None or price is None:
            account_value, price = self.get_crypto_account_value(crypto_model)
        return f'\n\t\t{crypto_model.symbol}:\t{account_value:.2f} =\t{account_value * price:.2f} (reward ratio: {reward_ratio:.2f})'
        
    def get_reward(self, action: Actions) -> float:
        net_return = (self.total_volume / self.initial_volume) - 1
        reward = net_return * 0.1  # Scale the reward as needed
        # reward = net_return * 3  # Scale the reward as needed
        asymmetry_factor = 2.0  # Adjust this factor to control asymmetry

        # Apply asymmetry: rewards positive returns more than penalizing negative returns
        if reward > 0:
            reward *= asymmetry_factor
            # reward += 0.5

        if len(self.past_volumes) >= self.short_term_reward_window:
            past_volume = np.sum(self.past_volumes[-self.short_term_reward_window:])/self.short_term_reward_window
            # if self.total_volume > past_volume:
            short_term_gain = (self.total_volume - past_volume) / past_volume
            reward += short_term_gain * 5
        reward *= 5
                
        return reward 