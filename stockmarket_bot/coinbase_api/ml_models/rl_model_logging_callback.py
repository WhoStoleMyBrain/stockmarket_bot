from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RLModelLoggingCallback(BaseCallback):
    """
    A custom callback that logs the total volume, reward, and usdc values to TensorBoard.
    """
    def __init__(self, verbose: int = 0):
        super(RLModelLoggingCallback, self).__init__(verbose)

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

        # Log additional environment properties if needed
        # step_count = env.data_handler.get_step_count()
        past_volumes = env.unwrapped.data_handler.past_volumes

        self.logger.record('env/total_volume', total_volume)
        self.logger.record('env/usdc', usdc)
        self.logger.record('env/reward', reward)
        self.logger.record('env/past_volumes', np.mean(past_volumes))
        # self.logger.record('env/step_count', step_count)
        # self.logger.record('env/total_steps', total_steps)

        # Ensure the logs are written at every step
        self.logger.dump(self.num_timesteps)
        return True
