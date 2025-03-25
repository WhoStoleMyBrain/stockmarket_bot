from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import threading

class RLModelLoggingCallback(BaseCallback):
    def __init__(self, log_interval: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.total_reward = 0
        self.step_count = 0
        self.total_step_count = 0
        self.total_cost_for_action = 0
        self.writer = None
        self.log_queue = []
        self.lock = threading.Lock()
        self.phase = "default"  # new attribute to mark the current phase

    def set_phase(self, phase_str: str):
        """Set the current training phase marker."""
        self.phase = phase_str

    def _on_training_start(self) -> None:
        if self.total_reward is None:
            self.total_reward = 0
        if self.step_count is None:
            self.step_count = 0
        if self.total_step_count is None:
            self.total_step_count = 0
        if self.total_cost_for_action is None:
            self.total_cost_for_action = 0
        output_formats = self.logger.output_formats
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_training_end(self):
        reward = self.locals.get('rewards', [0])[0]
        # self.total_reward += reward
        env = self.training_env.envs[0]
        # self.total_cost_for_action += env.unwrapped.data_handler.costs_for_action
        state = env.unwrapped.state
        timestamp = env.unwrapped.data_handler.timestamp
        current_price = env.unwrapped.data_handler.get_crypto_value_from_cache(env.unwrapped.data_handler.crypto.symbol, timestamp)
        total_volume = state[0]
        usdc = state[1]
        avg_reward = self.total_reward / self.step_count
        initial_crypto_price = env.unwrapped.data_handler.initial_crypto_price
        crypto_price = current_price
        baseline_value = env.unwrapped.data_handler.initial_volume * (crypto_price / float(initial_crypto_price))
        buy_hold_ratio = total_volume / baseline_value
        portfolio_return = (total_volume / env.unwrapped.data_handler.initial_volume)
        portfolio_strength = portfolio_return / buy_hold_ratio
        winning_trades = getattr(env.unwrapped.data_handler, "winning_trades", 0)
        losing_trades = getattr(env.unwrapped.data_handler, "losing_trades", 0)
        if (winning_trades + losing_trades) > 0:
            win_percentage = winning_trades / (winning_trades + losing_trades)
        else:
            win_percentage = 0.0
        actions_per_day = env.unwrapped.data_handler.current_daily_trades
        
        gs = self.total_step_count
        self.tb_formatter.writer.add_scalar("total_sparse/total_volume", total_volume, gs)
        self.tb_formatter.writer.add_scalar("total_sparse/usdc", usdc, gs)
        self.tb_formatter.writer.add_scalar("total_sparse/reward", reward, gs)
        self.tb_formatter.writer.add_scalar("total_sparse/average_reward", avg_reward, gs)
        self.tb_formatter.writer.add_scalar("total_sparse/total_cost_for_action", self.total_cost_for_action, gs)
        self.tb_formatter.writer.add_scalar("total_sparse/buy_hold_ratio", buy_hold_ratio, gs)
        self.tb_formatter.writer.add_scalar("total_sparse/portfolio_return", portfolio_return, gs)
        self.tb_formatter.writer.add_scalar("total_sparse/win_percentage", win_percentage, gs)
        self.tb_formatter.writer.add_scalar("total_sparse/portfolio_strength", portfolio_strength, gs)
        self.tb_formatter.writer.add_scalar("total_sparse/actions_per_day", actions_per_day, gs)
        self.logger.dump(self.num_timesteps)
        return super()._on_training_end()
            
    
    def _on_rollout_end(self):
        self.logger.dump(self.num_timesteps)
        return super()._on_rollout_end()

    def reset(self):
        self.total_reward = 0
        self.total_cost_for_action = 0
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        self.total_step_count += 1
        reward = self.locals.get('rewards', [0])[0]
        self.total_reward += reward
        env = self.training_env.envs[0]
        self.total_cost_for_action += env.unwrapped.data_handler.costs_for_action
        if self.step_count % self.log_interval == 0:
            state = env.unwrapped.state
            timestamp = env.unwrapped.data_handler.timestamp
            current_price = env.unwrapped.data_handler.get_crypto_value_from_cache(env.unwrapped.data_handler.crypto.symbol, timestamp)
            total_volume = state[0]
            usdc = state[1]
            avg_reward = self.total_reward / self.step_count
            initial_crypto_price = env.unwrapped.data_handler.initial_crypto_price
            crypto_price = current_price
            baseline_value = env.unwrapped.data_handler.initial_volume * (crypto_price / float(initial_crypto_price))
            buy_hold_ratio = total_volume / baseline_value
            portfolio_return = (total_volume / env.unwrapped.data_handler.initial_volume)
            portfolio_strength = portfolio_return / buy_hold_ratio
            winning_trades = getattr(env.unwrapped.data_handler, "winning_trades", 0)
            losing_trades = getattr(env.unwrapped.data_handler, "losing_trades", 0)
            if (winning_trades + losing_trades) > 0:
                win_percentage = winning_trades / (winning_trades + losing_trades)
            else:
                win_percentage = 0.0
            actions_per_day = env.unwrapped.data_handler.current_daily_trades
            
            ls = self.step_count
            gs = self.total_step_count
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/total_volume", total_volume, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/usdc", usdc, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/reward", reward, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/average_reward", avg_reward, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/total_cost_for_action", self.total_cost_for_action, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/buy_hold_ratio", buy_hold_ratio, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/portfolio_return", portfolio_return, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/win_percentage", win_percentage, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/portfolio_strength", portfolio_strength, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/actions_per_day", actions_per_day, ls)
            
            self.tb_formatter.writer.add_scalar("total/total_volume", total_volume, gs)
            self.tb_formatter.writer.add_scalar("total/usdc", usdc, gs)
            self.tb_formatter.writer.add_scalar("total/reward", reward, gs)
            self.tb_formatter.writer.add_scalar("total/average_reward", avg_reward, gs)
            self.tb_formatter.writer.add_scalar("total/total_cost_for_action", self.total_cost_for_action, gs)
            self.tb_formatter.writer.add_scalar("total/buy_hold_ratio", buy_hold_ratio, gs)
            self.tb_formatter.writer.add_scalar("total/portfolio_return", portfolio_return, gs)
            self.tb_formatter.writer.add_scalar("total/win_percentage", win_percentage, gs)
            self.tb_formatter.writer.add_scalar("total/portfolio_strength", portfolio_strength, gs)
            self.tb_formatter.writer.add_scalar("total/actions_per_day", actions_per_day, gs)
            self.logger.dump(self.num_timesteps)
        return True
