from stable_baselines3.common.callbacks import BaseCallback

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
        # Log the desired values to TensorBoard
        total_volume = self.training_env.get_attr('state')[0][0]
        usdc = self.training_env.get_attr('state')[0][1]
        reward = self.training_env.get_attr('reward')

        self.logger.record('env/total_volume', total_volume)
        self.logger.record('env/reward', reward)
        self.logger.record('env/usdc', usdc)
        # Ensure the logs are written at every step
        self.logger.dump(self.num_timesteps)
        return True
