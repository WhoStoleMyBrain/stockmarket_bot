from coinbase_api.constants import crypto_models
import os 
from datetime import timedelta 
import numpy as np 
import pandas as pd 
from django.core.management.base import BaseCommand

from coinbase_api.enums import ExportFolder
from coinbase_api.models.models import AbstractOHLCV

class Command(BaseCommand): 
    help = 'Analyze crypto trades for a given crypto symbol within a specified time range.'
    
    
    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str,
                            help="Crypto currency symbol (e.g. BTC, ETH)")
        parser.add_argument('--start_timestep', type=int,
                            help="Starting point (in timesteps)")
        parser.add_argument('--duration', type=int,
                            help="Duration (in timesteps) over which to analyze")

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting..."))
        symbol = options['symbol'].upper()
        start_timestep = options['start_timestep']
        duration = options['duration']

        # Find the correct crypto model.
        crypto = next((cm for cm in crypto_models if cm.symbol.upper() == symbol), None)
        if crypto is None:
            self.stderr.write(self.style.ERROR(f"Crypto model for symbol '{symbol}' not found."))
            return
        self.stdout.write(self.style.SUCCESS(f"Fetched crypto model: {crypto.symbol}..."))
        # Load historical data.
        try:
            df_dict = self.fetch_all_historical_data_from_file(crypto, start_timestep)
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error loading historical data: {e}"))
            return
        self.stdout.write(self.style.SUCCESS("Loaded historical data..."))
        
        if symbol not in df_dict:
            self.stderr.write(self.style.ERROR(f"No data available for '{symbol}'."))
            return

        df = df_dict[symbol]
        total_data_points = len(df)
        if start_timestep < 0 or start_timestep >= total_data_points:
            self.stderr.write(self.style.ERROR("Starting timestep is out of data range."))
            return
        start_timestep = start_timestep * 12 #! input is days, not minutes.
        # Slice the DataFrame for the provided range.
        end_index = start_timestep + duration
        self.stdout.write(self.style.SUCCESS(f"Starting at: {start_timestep}. Finish at: {end_index}. Total: {total_data_points}"))
        df_slice = df.iloc[start_timestep:end_index].copy()
        if df_slice.empty:
            self.stderr.write(self.style.ERROR("No data in the specified range."))
            return
        
        self.stdout.write(self.style.SUCCESS(f"Starting at: {df_slice.index.min()}. Finish at: {df_slice.index.max()}"))

        # (a) Overall profitability: count and percentage of profitable trades.
        total_trades = len(df_slice)
        profitable_trades = (df_slice['trade_outcome'] == 1).sum()
        percentage_profitable = (profitable_trades / total_trades) * 100 if total_trades else 0

        self.stdout.write("---------- Overall Profitability ----------")
        self.stdout.write(f"Total trades analyzed: {total_trades}")
        self.stdout.write(f"Number of profitable trades: {profitable_trades}")
        self.stdout.write(f"Percentage of profitable trades: {percentage_profitable:.2f}%")

        # (b) Best-case scenario simulation.
        best_case = self.simulate_best_case(df, start_timestep, duration)
        if best_case:
            (num_best_trades, total_profit, win_rate,
            assumed_monthly_profit, avg_trade_duration) = best_case

            self.stdout.write("\n---------- Best-Case Scenario Simulation ----------")
            self.stdout.write(f"Total profitable trades executed: {num_best_trades}")
            self.stdout.write(f"Total profit percentage: {total_profit*100:.2f}%")
            self.stdout.write(f"Percentage of profitable trades: {win_rate*100:.2f}%")
            self.stdout.write(f"Assumed profit in a month: {assumed_monthly_profit*100:.2f}%")
            self.stdout.write(f"Average trade duration (timesteps): {avg_trade_duration:.2f}")
        else:
            self.stdout.write("\nBest-case scenario simulation could not be performed due to insufficient data.")

        # (c) Additional metrics.
        additional = self.compute_additional_metrics(df_slice)
        self.stdout.write("\n---------- Additional Metrics ----------")
        for key, value in additional.items():
            self.stdout.write(f"{key}: {value}")

    def fetch_all_historical_data_from_file(self, crypto: AbstractOHLCV, start_timestep: int):
        """
        Load historical data for the crypto from a Parquet file.
        The file is expected to be at <EXPORT_FOLDER>/<symbol>.parquet.
        If the file is not found, an exception is raised.
        """
        file_path = os.path.join(ExportFolder.EXPORT_FOLDER.value, f"{crypto.symbol}.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Historical data file for {crypto.symbol} not found at {file_path}")
        try:
            # start_timestamp = self.timestamp - timedelta(minutes=(self.lstm_sequence_length - 1) * 5)
            # end_timestamp = self.timestamp + timedelta(minutes=self.total_steps * 5)
            
            # Option 2: Read the full file and filter in-memory.
            df = pd.read_parquet(file_path)
        except Exception as e:
            raise Exception(f"Error reading Parquet file: {e}")
        self.stdout.write(self.style.SUCCESS("Read parquet file..."))
        # Ensure the DataFrame has a datetime index.
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            self.stdout.write(self.style.SUCCESS("Added timestamp column..."))
        df.sort_index(inplace=True)
        df.fillna(0, inplace=True)
        df = self.add_trade_outcome_column(df)
        self.stdout.write(self.style.SUCCESS("Added trade outcome..."))
        # Build a mapping from crypto symbol to its DataFrame.
        self.timestamp_to_index = {ts: idx for idx, ts in enumerate(df.index)}
        self.stdout.write(self.style.SUCCESS("Timestamp to index dict created..."))
        return {crypto.symbol.upper(): df}

    def add_trade_outcome_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a new column 'trade_outcome' to the DataFrame.
        For each row, if buying at that row’s close price would eventually
        result in a profit (i.e. the 'close' price hits 103% before falling to 99%),
        assigns 1; if it would result in a loss (99% reached first), assigns -1;
        if it cannot be determined, assigns 0.
        """
        outcomes = np.zeros(len(df), dtype=int)
        close_vals = df['close'].values
        n = len(close_vals)
        # For each row, search ahead to see if the profit or loss threshold is hit.
        for i in range(n):
            p0 = close_vals[i]
            upper = p0 * 1.03
            lower = p0 * 0.99
            outcome = 0
            for j in range(i + 1, n):
                if close_vals[j] >= upper:
                    outcome = 1
                    break
                elif close_vals[j] <= lower:
                    outcome = -1
                    break
            outcomes[i] = outcome
        df['trade_outcome'] = outcomes
        return df

    def simulate_best_case(self, df: pd.DataFrame, start_timestep: int, duration: int):
        """
        Simulate a best-case scenario where:
        - You start at the given timestep.
        - At the first instance where a profitable trade (trade_outcome == 1) is possible,
            you “enter” the trade and then wait until the trade meets the 3% profit threshold.
        - Immediately after exiting the trade, you look for the next profitable opportunity.
        Returns a tuple with:
            (number_of_trades, total_profit, win_rate, assumed_monthly_profit, avg_trade_duration)
        """
        end_timestep = start_timestep + duration
        n = len(df)
        if start_timestep >= n:
            return None

        current_index = start_timestep
        trade_profits = []
        trade_durations = []

        # Iterate until the end_timestep or the end of available data.
        while current_index < end_timestep and current_index < n:
            entry_row = df.iloc[current_index]
            entry_price = entry_row['close']
            if entry_row['trade_outcome'] != 1:
                current_index += 1
                continue
            target_price = entry_price * 1.03
            exit_index = None
            # Search for the first timestep where the 3% target is met.
            for j in range(current_index + 1, min(n, end_timestep)):
                if df.iloc[j]['close'] >= target_price:
                    exit_index = j
                    break
            if exit_index is not None:
                duration_trade = exit_index - current_index
                trade_durations.append(duration_trade)
                # Assume a 3% profit (i.e. 0.03).
                trade_profits.append(0.03)
                # Move immediately after the exit for next trade.
                current_index = exit_index + 1
            else:
                # If no profitable exit is found, break out.
                break

        if not trade_profits:
            return None

        num_trades = len(trade_profits)
        total_profit = 1.03 ** len(trade_profits) - 1
        win_rate = 1.0  # All trades in this simulation are profitable by assumption.
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0

        # Extrapolate an assumed monthly profit.
        # Each timestep is 5 minutes.
        simulation_minutes = duration * 5
        minutes_in_month = 30 * 24 * 60  # Approximation for a 30-day month.
        assumed_monthly_profit = (total_profit / simulation_minutes) * minutes_in_month
        return num_trades, total_profit, win_rate, assumed_monthly_profit, avg_trade_duration

    def compute_additional_metrics(self, df: pd.DataFrame):
        """
        Compute additional metrics for trade analysis:
        - Total trades attempted.
        - Count and percentage of losing trades.
        - Count of neutral (undetermined) trades.
        - Average trade duration: approximated as the number of timesteps until an outcome is determined.
        """
        total_trades = len(df)
        losing_trades = (df['trade_outcome'] == -1).sum()
        winning_trades = (df['trade_outcome'] == 1).sum()
        neutral_trades = (df['trade_outcome'] == 0).sum()

        # Estimate average trade duration by scanning each row to determine when a trade result is reached.
        durations = []
        close_vals = df['close'].values
        n = len(close_vals)
        for i in range(n):
            p0 = close_vals[i]
            upper = p0 * 1.03
            lower = p0 * 0.99
            duration_found = None
            for j in range(i + 1, n):
                if close_vals[j] >= upper or close_vals[j] <= lower:
                    duration_found = j - i
                    break
            if duration_found is not None:
                durations.append(duration_found)
        avg_duration = np.mean(durations) if durations else None

        metrics = {
            "Total trades in period": total_trades,
            "Winning trades count": winning_trades,
            "Losing trades count": losing_trades,
            "Neutral trades count": neutral_trades,
            "Average trade duration (timesteps)": avg_duration,
        }
        return metrics
