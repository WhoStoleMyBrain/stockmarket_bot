from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.logger import TensorBoardOutputFormat
from datetime import datetime
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

    def reset(self):
        self.total_reward = 0
        self.total_cost_for_action = 0
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        self.total_step_count += 1
        if self.step_count % self.log_interval == 0:
            env = self.training_env.envs[0]
            state = env.unwrapped.state
            total_volume = state[0]
            usdc = state[1]
            reward = self.locals.get('rewards', [0])[0] #! ?? this one?
            self.total_reward += reward
            avg_reward = self.total_reward / self.step_count
            self.total_cost_for_action += env.unwrapped.data_handler.costs_for_action
            
            # timestamp = env.unwrapped.data_handler.timestamp
            # ts_int = datetime.timestamp(timestamp)
            ls = self.step_count
            gs = self.total_step_count
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/total_volume", total_volume, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/usdc", usdc, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/reward", reward, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/average_reward", avg_reward, ls)
            self.tb_formatter.writer.add_scalar(f"phase/{self.phase}/total_cost_for_action", self.total_cost_for_action, ls)
            # Also log overall metrics (phase-independent) for an aggregated view.
            self.tb_formatter.writer.add_scalar("total/total_volume", total_volume, gs)
            self.tb_formatter.writer.add_scalar("total/usdc", usdc, gs)
            self.tb_formatter.writer.add_scalar("total/reward", reward, gs)
            self.tb_formatter.writer.add_scalar("total/average_reward", avg_reward, gs)
            self.tb_formatter.writer.add_scalar("total/total_cost_for_action", self.total_cost_for_action, gs)
        return True
