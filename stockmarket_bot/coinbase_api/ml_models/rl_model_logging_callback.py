from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.logger import TensorBoardOutputFormat
from datetime import datetime
import threading

class RLModelLoggingCallback(BaseCallback):
    def __init__(self, log_interval: int = 100, verbose: int = 0):
        super(RLModelLoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.total_reward = 0
        self.step_count = 0
        self.total_cost_for_action = 0
        self.writer = None
        self.log_queue = []
        self.lock = threading.Lock()

    def _on_training_start(self) -> None:
        output_formats = self.logger.output_formats
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        state = env.unwrapped.state
        total_volume = state[0]
        usdc = state[1]
        reward = self.locals.get('rewards', [0])[0]
        self.total_reward += reward
        self.step_count += 1
        avg_reward = self.total_reward / self.step_count
        self.total_cost_for_action += env.unwrapped.cost_for_action
        timestamp = env.unwrapped.data_handler.timestamp
        past_volumes = env.unwrapped.data_handler.past_volumes
        ts_int = datetime.timestamp(timestamp)

        log_data = {
            'timestamp': timestamp,
            'total_volume': total_volume,
            'usdc': usdc,
            'reward': reward,
            'avg_reward': avg_reward,
            'past_volumes': np.mean(past_volumes),
            'total_cost_for_action': self.total_cost_for_action / self.step_count,
            'ts_int': ts_int
        }

        with self.lock:
            self.log_queue.append(log_data)

        if self.step_count % self.log_interval == 0:
            self._flush_logs()

        return True

    def _flush_logs(self):
        with self.lock:
            logs_to_write = self.log_queue
            self.log_queue = []

        for log_data in logs_to_write:
            self.logger.record('env/timestamp', log_data['timestamp'])
            self.logger.record('env/total_volume', log_data['total_volume'])
            self.logger.record('env/usdc', log_data['usdc'])
            self.logger.record('env/reward', log_data['reward'])
            self.logger.record('env/average_reward', log_data['avg_reward'])
            self.logger.record('env/past_volumes', log_data['past_volumes'])
            self.logger.record('env/total_cost_for_action', log_data['total_cost_for_action'])
            
            self.tb_formatter.writer.add_scalar("train/total_volume", log_data['total_volume'], log_data['ts_int'])
            self.tb_formatter.writer.add_scalar('train/total_volume', log_data['total_volume'], log_data['ts_int'])
            self.tb_formatter.writer.add_scalar('train/usdc', log_data['usdc'], log_data['ts_int'])
            self.tb_formatter.writer.add_scalar('train/reward', log_data['reward'], log_data['ts_int'])
            self.tb_formatter.writer.add_scalar('train/past_volumes', log_data['past_volumes'], log_data['ts_int'])
            self.tb_formatter.writer.add_scalar('train/total_cost_for_action', log_data['total_cost_for_action'], log_data['ts_int'])
            self.tb_formatter.writer.add_scalar('train/average_reward', log_data['avg_reward'], log_data['ts_int'])

        self.tb_formatter.writer.flush()
        self.logger.dump(self.num_timesteps)

    def _on_training_end(self) -> None:
        self._flush_logs()

    def _on_rollout_end(self) -> None:
        self._flush_logs()
