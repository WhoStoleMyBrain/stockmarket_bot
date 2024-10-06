from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.logger import TensorBoardOutputFormat
from datetime import datetime

class ValidationLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(ValidationLoggingCallback, self).__init__(verbose)
        self.total_reward = 0
        self.step_count = 0
        self.total_cost_for_action = 0
        self.writer = None
        self.log_dir = log_dir

    def _on_training_start(self) -> None:
        output_formats = self.logger.output_formats
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        state = env.unwrapped.state

        # Core environment-related metrics
        total_volume = state[0]  # Total value of the portfolio
        usdc = state[1]  # Amount of USDC held
        reward = self.locals.get('rewards', [0])[0]  # Immediate reward for the action
        self.total_reward += reward
        self.step_count += 1
        avg_reward = self.total_reward / self.step_count  # Average reward per step
        self.total_cost_for_action += env.unwrapped.cost_for_action  # Total cost incurred for actions
        timestamp = env.unwrapped.data_handler.timestamp  # Current timestamp in the environment
        past_volumes = env.unwrapped.data_handler.past_volumes  # Historical volumes for the portfolio

        # Calculate additional metrics
        ts_int = datetime.timestamp(timestamp)  # Convert timestamp to a Unix timestamp

        # New metrics
        max_drawdown = self.calculate_max_drawdown(past_volumes)  # Max drawdown observed in the past volumes
        volatility = self.calculate_volatility(past_volumes)  # Volatility of the portfolio over time
        sharpe_ratio = self.calculate_sharpe_ratio(past_volumes, avg_reward)  # Sharpe ratio for risk-adjusted returns

        # Log all relevant metrics to TensorBoard
        self.logger.record('validation/timestamp', timestamp)
        self.logger.record('validation/total_volume', total_volume)
        self.logger.record('validation/usdc', usdc)
        self.logger.record('validation/reward', reward)
        self.logger.record('validation/average_reward', avg_reward)
        self.logger.record('validation/past_volumes', np.mean(past_volumes))
        self.logger.record('validation/total_cost_for_action', self.total_cost_for_action / self.step_count)
        self.logger.record('validation/max_drawdown', max_drawdown)
        self.logger.record('validation/volatility', volatility)
        self.logger.record('validation/sharpe_ratio', sharpe_ratio)
        
        # Write metrics to TensorBoard
        self.tb_formatter.writer.add_scalar("validation/total_volume", total_volume, ts_int)
        self.tb_formatter.writer.add_scalar('validation/usdc', usdc, ts_int)
        self.tb_formatter.writer.add_scalar('validation/reward', reward, ts_int)
        self.tb_formatter.writer.add_scalar('validation/average_reward', avg_reward, ts_int)
        self.tb_formatter.writer.add_scalar('validation/past_volumes', np.mean(past_volumes), ts_int)
        self.tb_formatter.writer.add_scalar('validation/total_cost_for_action', self.total_cost_for_action / self.step_count, ts_int)
        self.tb_formatter.writer.add_scalar('validation/max_drawdown', max_drawdown, ts_int)
        self.tb_formatter.writer.add_scalar('validation/volatility', volatility, ts_int)
        self.tb_formatter.writer.add_scalar('validation/sharpe_ratio', sharpe_ratio, ts_int)

        self.tb_formatter.writer.flush()
        self.logger.dump(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        self._flush_logs()

    def _on_rollout_end(self) -> None:
        self._flush_logs()

    def _flush_logs(self):
        self.tb_formatter.writer.flush()
        self.logger.dump(self.num_timesteps)

    def calculate_max_drawdown(self, volumes):
        """
        Calculate the maximum drawdown, which is the largest peak-to-trough decline.
        """
        if len(volumes) < 2:
            return 0
        peak = volumes[0]
        max_drawdown = 0
        for volume in volumes:
            if volume > peak:
                peak = volume
            drawdown = (peak - volume) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def calculate_volatility(self, volumes):
        """
        Calculate volatility as the standard deviation of the percentage changes in volume.
        """
        if len(volumes) < 2:
            return 0
        returns = np.diff(volumes) / volumes[:-1]
        return np.std(returns)

    def calculate_sharpe_ratio(self, volumes, avg_reward):
        """
        Calculate the Sharpe ratio, assuming a risk-free rate of 0.
        """
        if len(volumes) < 2:
            return 0
        returns = np.diff(volumes) / volumes[:-1]
        if np.std(returns) == 0:
            return 0
        return avg_reward / np.std(returns)
