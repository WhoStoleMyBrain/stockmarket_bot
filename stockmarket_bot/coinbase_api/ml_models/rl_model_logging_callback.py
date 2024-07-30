from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
# import tensorflow as tf
from stable_baselines3.common.logger import TensorBoardOutputFormat
from datetime import datetime

class RLModelLoggingCallback(BaseCallback):
    """
    A custom callback that logs various metrics to TensorBoard.
    """
    def __init__(self, verbose: int = 0):
        super(RLModelLoggingCallback, self).__init__(verbose)
        self.total_reward = 0
        self.step_count = 0
        self.total_cost_for_action = 0
        self.writer = None
        
    def _on_training_start(self) -> None:
        # self.writer = tf.summary.create_file_writer(self.logger.dir)
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

        
    # def _on_training_end(self) -> None:
    #     """
    #     This method is called after the last rollout ends.
    #     """
    #     if self.writer:
    #         self.writer.close()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Retrieve the environment
        env = self.training_env.envs[0]
        
        # Get the current state from the environment
        state = env.unwrapped.state

        # Extract the total volume and usdc from the state
        total_volume = state[0]
        usdc = state[1]

        # Log the reward separately as it is returned by the environment step
        reward = self.locals.get('rewards', [0])[0]
        self.total_reward += reward
        self.step_count += 1

        # Calculate average reward
        avg_reward = self.total_reward / self.step_count

        self.total_cost_for_action += env.unwrapped.cost_for_action
        # Extract the timestamp from the environment
        timestamp = env.unwrapped.data_handler.timestamp

        # Log additional environment properties
        past_volumes = env.unwrapped.data_handler.past_volumes
        # account_holdings = env.unwrapped.data_handler.account_holdings
        # transaction_costs = env.unwrapped.data_handler.transaction_costs
        # liquidity = env.unwrapped.get_liquidity()

        # Log metrics to TensorBoard
        self.logger.record('env/timestamp', timestamp)
        self.logger.record('env/total_volume', total_volume)
        self.logger.record('env/usdc', usdc)
        self.logger.record('env/reward', reward)
        self.logger.record('env/average_reward', avg_reward)
        self.logger.record('env/past_volumes', np.mean(past_volumes))
        # self.logger.record('env/step_count', self.step_count)
        # self.logger.record('env/account_holdings', np.mean(account_holdings))
        self.logger.record('env/total_cost_for_action', self.total_cost_for_action / self.step_count)
        # self.logger.record('env/liquidity', liquidity)
        # if (self.writer is not None):
            # with self.writer.as_default():
        ts_int = datetime.timestamp(timestamp)
        
        self.tb_formatter.writer.add_scalar("train/total_volume", total_volume, ts_int)
        self.tb_formatter.writer.add_scalar('train/total_volume', total_volume, ts_int)
        self.tb_formatter.writer.add_scalar('train/usdc', usdc, ts_int)
        self.tb_formatter.writer.add_scalar('train/reward', reward, ts_int)
        self.tb_formatter.writer.add_scalar('train/past_volumes', np.mean(past_volumes), ts_int)
        self.tb_formatter.writer.add_scalar('train/total_cost_for_action', self.total_cost_for_action / self.step_count, ts_int)
        self.tb_formatter.writer.add_scalar('train/average_reward', avg_reward, ts_int)
        self.tb_formatter.writer.flush()
        # else:
        #     print('self.writer was none')
        # Ensure the logs are written at every step
        self.logger.dump(self.num_timesteps)
        return True
