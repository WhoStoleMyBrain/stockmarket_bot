import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class DynamicObservationEnv(gym.Env):
    def __init__(self, observation_shape):
        super(DynamicObservationEnv, self).__init__()
        self.observation_shape = observation_shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_shape,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # Example action space with 3 discrete actions

    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.random.random(self.observation_shape).astype(np.float32)
        reward = np.random.random()
        done = np.random.rand() > 0.95  # Randomly end the episode
        info = {}
        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

print('starting up...')
# Example usage
initial_observation_shape = 10
env = DynamicObservationEnv(observation_shape=initial_observation_shape)
print('created first env...')
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
print('learned model once. saving now...')

# Save the model
model.save("dynamic_observation_model")

# Create a new environment with a different observation space
print('building new env...')
new_observation_shape = 20
new_env = DynamicObservationEnv(observation_shape=new_observation_shape)
print('loading model...')
# Load the model
model = PPO.load("dynamic_observation_model")

# Manually update the observation space
model.observation_space = new_env.observation_space
model.policy.observation_space = new_env.observation_space

# Re-wrap the environment
model.set_env(new_env)

print('learning model again...')
# Continue training
model.learn(total_timesteps=1000)
print('finished learning...')