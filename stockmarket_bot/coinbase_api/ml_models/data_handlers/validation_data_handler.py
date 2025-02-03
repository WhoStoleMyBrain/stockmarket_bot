import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class ValidationDataHandler:
    """
    A data handler specifically designed for validation scenarios.
    This will allow the separation of training and validation logic.
    """
    
    def __init__(self, scenario_config: dict, initial_volume: float = 1000) -> None:
        self.initial_volume = initial_volume
        self.total_steps = scenario_config.get('total_steps', 1024)
        self.scenario_config = scenario_config
        self.current_scenario = None
        self.timestamp = datetime.now()  # Starting timestamp
        self.state = self.reset_state()
        self.scenarios = scenario_config.get('scenarios', [])

    def load_scenario(self, scenario_name: str) -> None:
        """
        Loads the configuration for a specific validation scenario.
        """
        for scenario in self.scenarios:
            if scenario['name'] == scenario_name:
                self.current_scenario = scenario
                self.timestamp = scenario.get('start_time', datetime.now())
                return
        raise ValueError(f"Scenario {scenario_name} not found in config")

    def reset_state(self):
        """
        Resets the environment state based on the initial volume and starting timestamp.
        """
        self.state = {'volume': self.initial_volume, 'holdings': {}}
        return self.state

    def generate_linear_gain(self, slope: float) -> None:
        """
        Generates linear gain over time for validation.
        """
        self.state['volume'] += slope
        self.timestamp += timedelta(hours=1)

    def generate_exponential_gain(self, growth_rate: float) -> None:
        """
        Generates exponential gain over time for validation.
        """
        self.state['volume'] *= (1 + growth_rate)
        self.timestamp += timedelta(hours=1)

    def update_state(self, action):
        """
        Simulates the action taken by the model during validation.
        Here, we process the action and update the state accordingly.
        """
        # Update the state based on the action and scenario
        if self.current_scenario['type'] == 'linear_gain':
            self.generate_linear_gain(self.current_scenario['params']['slope'])
        elif self.current_scenario['type'] == 'exponential_gain':
            self.generate_exponential_gain(self.current_scenario['params']['growth_rate'])
        else:
            raise NotImplementedError("Scenario type not implemented")

        # Process the action
        # Example: implement hold, buy, sell logic based on action
        # Update holdings, apply transaction costs, etc.

        return self.state
    
    def generate_linear_data(self, timestamps, slope, base_value=1000) -> pd.DataFrame:
        """
        Generate synthetic data with a linear trend.
        """
        length = len(timestamps)
        data = {
            'close': np.linspace(base_value, base_value + slope * length, length),
            'volume': np.ones(length) * 1000,  # Example: constant volume
            'sma': np.linspace(base_value, base_value + slope * length, length),
            'ema': np.linspace(base_value, base_value + slope * length, length),
            'rsi': np.ones(length) * 50,  # Example: constant RSI
            'macd': np.linspace(-5, 5, length),
            'bollinger_high': np.linspace(base_value + 5, base_value + slope * length + 5, length),
            'bollinger_low': np.linspace(base_value - 5, base_value + slope * length - 5, length),
            'vmap': np.linspace(base_value, base_value + slope * length, length),
            'percentage_returns': np.zeros(length),
            'log_returns': np.zeros(length)
        }
        return pd.DataFrame(data)

    def generate_exponential_data(self, timestamps, growth_rate, base_value=1000) -> pd.DataFrame:
        """
        Generate synthetic data with an exponential trend.
        """
        length = len(timestamps)
        data = {
            'close': base_value * np.exp(growth_rate * np.arange(length)),
            'volume': np.ones(length) * 1000,
            'sma': base_value * np.exp(growth_rate * np.arange(length)),
            'ema': base_value * np.exp(growth_rate * np.arange(length)),
            'rsi': np.ones(length) * 50,
            'macd': np.linspace(-5, 5, length),
            'bollinger_high': base_value * np.exp(growth_rate * np.arange(length)) + 5,
            'bollinger_low': base_value * np.exp(growth_rate * np.arange(length)) - 5,
            'vmap': base_value * np.exp(growth_rate * np.arange(length)),
            'percentage_returns': np.zeros(length),
            'log_returns': np.zeros(length)
        }
        return pd.DataFrame(data)

    def generate_constant_data(self, timestamps, constant_value=1000) -> pd.DataFrame:
        """
        Generate synthetic data with constant market conditions.
        """
        length = len(timestamps)
        data = {
            'close': np.ones(length) * constant_value,
            'volume': np.ones(length) * 1000,
            'sma': np.ones(length) * constant_value,
            'ema': np.ones(length) * constant_value,
            'rsi': np.ones(length) * 50,
            'macd': np.zeros(length),
            'bollinger_high': np.ones(length) * (constant_value + 5),
            'bollinger_low': np.ones(length) * (constant_value - 5),
            'vmap': np.ones(length) * constant_value,
            'percentage_returns': np.zeros(length),
            'log_returns': np.zeros(length)
        }
        return pd.DataFrame(data)

    def generate_volatile_data(self, timestamps, volatility_level=0.05, base_value=1000) -> pd.DataFrame:
        """
        Generate synthetic data with random volatility (Gaussian noise).
        """
        length = len(timestamps)
        random_walk = np.cumsum(np.random.normal(0, volatility_level, length)) + base_value
        data = {
            'close': random_walk,
            'volume': np.ones(length) * 1000,
            'sma': pd.Series(random_walk).rolling(window=10, min_periods=1).mean(),
            'ema': pd.Series(random_walk).ewm(span=10, adjust=False).mean(),
            'rsi': np.ones(length) * 50,
            'macd': np.random.normal(0, 2, length),
            'bollinger_high': random_walk + 5,
            'bollinger_low': random_walk - 5,
            'vmap': random_walk,
            'percentage_returns': np.zeros(length),
            'log_returns': np.zeros(length)
        }
        return pd.DataFrame(data)

    def generate_bear_market_data(self, timestamps, slope, base_value=1000) -> pd.DataFrame:
        """
        Generate synthetic data with a steady decline (bear market).
        """
        return self.generate_linear_data(timestamps, -abs(slope), base_value)

    def generate_market_crash_data(self, timestamps, crash_time, severity, recovery_rate, base_value=1000) -> pd.DataFrame:
        """
        Generate synthetic data simulating a market crash followed by recovery.
        """
        length = len(timestamps)
        data = np.ones(length) * base_value

        # Apply crash at the specified time
        data[crash_time:] = data[crash_time] * (1 - severity)
        # Apply recovery after the crash
        recovery_start = crash_time + 1
        data[recovery_start:] = data[recovery_start] * np.exp(recovery_rate * np.arange(length - recovery_start))

        df = pd.DataFrame({
            'close': data,
            'volume': np.ones(length) * 1000,
            'sma': pd.Series(data).rolling(window=10, min_periods=1).mean(),
            'ema': pd.Series(data).ewm(span=10, adjust=False).mean(),
            'rsi': np.ones(length) * 50,
            'macd': np.zeros(length),
            'bollinger_high': data + 5,
            'bollinger_low': data - 5,
            'vmap': data,
            'percentage_returns': np.zeros(length),
            'log_returns': np.zeros(length)
        })
        return df

    def generate_historical_data(self, start_date, end_date) -> pd.DataFrame:
        """
        Use real historical data between the given start and end dates.
        """
        # Placeholder - Replace with actual historical data fetching logic
        # You would query your historical database for data between start_date and end_date
        raise NotImplementedError("Historical data fetching is not implemented yet.")

    def generate_multi_year_data(self, start_date, end_date, transaction_costs=0.0) -> pd.DataFrame:
        """
        Generate long-term synthetic data for a multi-year simulation.
        """
        # Use a long-term linear or exponential trend as a placeholder
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        return self.generate_linear_data(timestamps, slope=0.001, base_value=1000)

    def generate_stress_test_data(self, timestamps, volatility, spread, liquidity) -> pd.DataFrame:
        """
        Generate synthetic data for stress-testing the model under extreme conditions.
        """
        length = len(timestamps)
        random_walk = np.cumsum(np.random.normal(0, volatility, length)) + 1000
        data = {
            'close': random_walk,
            'volume': np.random.randint(1, liquidity, length),
            'sma': pd.Series(random_walk).rolling(window=10, min_periods=1).mean(),
            'ema': pd.Series(random_walk).ewm(span=10, adjust=False).mean(),
            'rsi': np.ones(length) * 50,
            'macd': np.random.normal(0, 2, length),
            'bollinger_high': random_walk + spread,
            'bollinger_low': random_walk - spread,
            'vmap': random_walk,
            'percentage_returns': np.zeros(length),
            'log_returns': np.zeros(length)
        }
        return pd.DataFrame(data)

    def generate_edge_case_data(self, timestamps, scenario='near_zero_volume') -> pd.DataFrame:
        """
        Generate synthetic data for edge case scenarios such as near-zero volume or sudden price spikes.
        """
        length = len(timestamps)
        if scenario == 'near_zero_volume':
            volume = np.zeros(length)
        elif scenario == 'sudden_spike':
            price = np.ones(length) * 1000
            spike_time = random.randint(0, length - 1)
            price[spike_time:] = price[spike_time] * 2  # sudden price spike
        else:
            raise ValueError(f"Unknown edge case scenario: {scenario}")

        data = {
            'close': price if scenario == 'sudden_spike' else np.ones(length) * 1000,
            'volume': volume if scenario == 'near_zero_volume' else np.ones(length) * 1000,
            'sma': np.ones(length) * 1000,
            'ema': np.ones(length) * 1000,
            'rsi': np.ones(length) * 50,
            'macd': np.zeros(length),
            'bollinger_high': np.ones(length) * 1005,
            'bollinger_low': np.ones(length) * 995,
            'vmap': np.ones(length) * 1000,
            'percentage_returns': np.zeros(length),
            'log_returns': np.zeros(length)
        }
        return pd.DataFrame(data)

    def log_metrics(self, step):
        """
        Logs the metrics at each step for TensorBoard.
        """
        metrics = {
            'volume': self.state['volume'],
            'timestamp': self.timestamp,
            # Add other metrics to track here (e.g., reward, holdings, etc.)
        }
        # Implement logic to write these metrics to TensorBoard
        return metrics
