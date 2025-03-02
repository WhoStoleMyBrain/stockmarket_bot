from django.core.management.base import BaseCommand
from stable_baselines3 import PPO
from coinbase_api.ml_models.RL_decider_model import CustomEnv
from coinbase_api.ml_models.data_handlers.validation_data_handler import ValidationDataHandler  # New Validation Handler
from coinbase_api.ml_models.custom_policy import CustomPolicy
from coinbase_api.ml_models.validation_logging_callback import ValidationLoggingCallback
import json
import os


class Command(BaseCommand):
    help = 'Validate the RL model using various synthetic data scenarios and log detailed metrics.'

    def add_arguments(self, parser):
        parser.add_argument('--config', type=str, help='Path to the validation config file', required=True)
        parser.add_argument('--training_folder', type=str, help='Path to the training folder', required=True)

    def handle(self, *args, **kwargs):
        config_path = kwargs['config']
        training_folder = kwargs['training_folder']

        # Load the config file
        config = self.load_config(config_path)

        # Paths for model and logging
        model_path = os.path.join(training_folder, 'rl_model.pkl')
        log_dir = os.path.join(training_folder, 'validation_logs')

        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Loop through all validation scenarios
        for scenario in config['scenarios']:
            self.run_scenario(model_path, scenario, log_dir)

    def load_config(self, config_path):
        """
        Load the JSON configuration for validation scenarios.
        """
        with open(config_path, 'r') as f:
            return json.load(f)

    def run_scenario(self, model_path, scenario, log_dir):
        """
        Run the validation for a specific scenario, loading the model and environment.
        """
        print(f"Running scenario: {scenario['name']}")

        # Create a validation data handler
        validation_data_handler = ValidationDataHandler(
            initial_volume=scenario.get('initial_volume', 1000),
            total_steps=scenario['total_steps'],
            scenario_config=scenario
        )

        # Initialize the environment with the validation data handler
        env = CustomEnv(data_handler=validation_data_handler)

        # Load the trained model
        model = PPO.load(model_path, env=env)

        # Create a log directory for this scenario
        scenario_log_dir = os.path.join(log_dir, scenario['name'])
        os.makedirs(scenario_log_dir, exist_ok=True)

        # Initialize the custom logging callback
        callback = ValidationLoggingCallback(log_dir=scenario_log_dir)

        # Run the validation
        model.learn(
            total_timesteps=scenario['total_steps'],
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"Validation_{scenario['name']}",
            log_interval=1,
            callback=[callback]
        )

        print(f"Scenario {scenario['name']} completed.")

