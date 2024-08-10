from django.core.management.base import BaseCommand, CommandError
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from coinbase_api.ml_models.RL_decider_model import CustomEnv
import traceback
import json
import shutil
from coinbase_api.ml_models.data_handlers.simulation_data_handler import SimulationDataHandler
from coinbase_api.ml_models.rl_model_logging_callback import RLModelLoggingCallback

CONFIG_FILE_PATH = 'coinbase_api/ml_models/training_configs/training_config_1.json'
ACTIVE_TRAININGS_PATH = 'coinbase_api/ml_models/active_trainings'
LOGS_BASE_DIR = '/logs'

class LinearSchedule:
    def __init__(self, initial_lr, final_lr, duration):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.duration = duration

    def __call__(self, progress_remaining):
        progress = 1 - progress_remaining
        return self.initial_lr + progress * (self.final_lr - self.initial_lr)
    
def get_clip_range(clip_range):
    def clip_range_fn(progress):
        return clip_range
    return clip_range_fn



class Command(BaseCommand):
    help = 'Train the RL model for a number of iterations over the whole dataset'

    def add_arguments(self, parser):
        parser.add_argument(
            '--config',
            type=str,
            help='Path to the training config file',
            default=CONFIG_FILE_PATH
        )
        parser.add_argument(
            '--continue_training',
            type=str,
            help='Name of the active training to continue',
            required=False
        )
    
    def __init__(self):
        super().__init__()
        self.model = None


    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def save_config(self, config, config_path):
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def handle(self, *args, **kwargs):
        config_path = kwargs.get('config', CONFIG_FILE_PATH)
        continue_training = kwargs.get('continue_training', None)

        if continue_training:
            active_training_path = os.path.join(ACTIVE_TRAININGS_PATH, continue_training)
            if not os.path.exists(active_training_path):
                raise CommandError(f'Active training {continue_training} does not exist.')
            config_path = os.path.join(active_training_path, 'current_config.json')
            model_path = os.path.join(active_training_path, 'rl_model.pkl')
            log_dir = os.path.join(LOGS_BASE_DIR, continue_training)
        else:
            config = self.load_config(config_path)
            training_name = os.path.splitext(os.path.basename(config_path))[0]
            existing_trainings = [d for d in os.listdir(ACTIVE_TRAININGS_PATH) if os.path.isdir(os.path.join(ACTIVE_TRAININGS_PATH, d)) and d.startswith(training_name)]
            new_index = len(existing_trainings)
            active_training_path = os.path.join(ACTIVE_TRAININGS_PATH, f"{training_name}_{new_index}")
            os.makedirs(active_training_path)
            shutil.copy(config_path, os.path.join(active_training_path, 'base_config.json'))
            config_path = os.path.join(active_training_path, 'current_config.json')
            self.save_config(config, config_path)
            model_path = os.path.join(active_training_path, 'rl_model.pkl')
            log_dir = os.path.join(LOGS_BASE_DIR, f"{training_name}_{new_index}")

        os.makedirs(log_dir, exist_ok=True)
        config = self.load_config(config_path)
        total_timesteps = config.get('total_timesteps', 1e6)
        phases = config.get('phases', [])
        current_stage = config.get('current_stage', 0)

        for phase_index, phase in enumerate(phases):
            if phase_index < current_stage:
                continue

            phase_timesteps = phase.get('phase_timesteps', int(phase['percentage'] * total_timesteps / 100))
            intervals = phase['intervals']
            interval_weights = phase['interval_weights']
            interval_transaction_costs = phase.get('interval_transaction_costs', 0.0)
            initial_lr = phase.get('initial_lr', 3e-4)
            final_lr = phase.get('final_lr', 1e-5)
            clip_range = phase.get('clip_range', 0.2)
            batch_size = phase.get('batch_size', 64)
            n_epochs = phase.get('n_epochs', 10)
            checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=f"{active_training_path}_checkpoint", name_prefix='rl_model_checkpoint')

            interval_list = [interval for interval, weight in zip(intervals, interval_weights) for _ in range(weight)]

            lr_schedule = LinearSchedule(config['phases'][current_stage].get('initial_lr', 3e-4), 
                                                config['phases'][current_stage].get('final_lr', 1e-5), 
                                                phase_timesteps)

            interval_index = 0
            while phase_timesteps > 0:
                interval = interval_list[interval_index % len(interval_list)]
                interval_index += 1

                print(f"Active Training Path: {active_training_path}")
                print(f"Phase Index: {phase_index}")
                print(f"Total Phases Length: {len(phases)}")
                print(f"Interval: {interval} (Index in Interval List: {interval_index}/{len(interval_list)})")
                print(f"Transaction Costs: {interval_transaction_costs}")
                print(f"Phase Timesteps: {phase_timesteps}")
                print(f"Initial LR: {initial_lr}")
                print(f"Final LR: {final_lr}")
                print(f"Clip Range: {clip_range}")
                print(f"Batch Size: {batch_size}")
                print(f"N Epochs: {n_epochs}")

                print(f'Starting training with interval: {interval}')
                if self.model is None or self.model.n_steps != interval:
                    if os.path.exists(model_path):
                        print('Loaded model!')
                        env = CustomEnv(data_handler=SimulationDataHandler(total_steps=interval, transaction_cost_factor=interval_transaction_costs))
                        self.model = PPO.load(model_path, env=env)
                        self.model.tensorboard_log = log_dir
                    else:
                        env = CustomEnv(data_handler=SimulationDataHandler(total_steps=interval, transaction_cost_factor=interval_transaction_costs))
                        self.model = PPO(
                            "MlpPolicy",
                            env=env,
                            verbose=0,
                            tensorboard_log=log_dir,
                            learning_rate=lr_schedule,
                            clip_range=get_clip_range(clip_range),
                            n_steps=interval,
                            batch_size=batch_size,
                            n_epochs=n_epochs
                        )
                    # Reset model's rollout buffer to match new n_steps
                    self.model.n_steps = interval
                    self.model._setup_model()
                else:
                    env = CustomEnv(data_handler=SimulationDataHandler(total_steps=interval, transaction_cost_factor=interval_transaction_costs))
                    self.model.set_env(env)
                    self.model.learning_rate = lr_schedule
                    self.model.clip_range = get_clip_range(clip_range)
                    self.model.n_steps = interval
                    self.model.batch_size = batch_size
                    self.model.n_epochs = n_epochs
                    self.model._setup_model()

                try:
                    self.model.learn(
                        total_timesteps=interval,
                        progress_bar=True,
                        reset_num_timesteps=False,
                        tb_log_name=f"ModelV1_{interval}_{interval_transaction_costs}",
                        log_interval=1,
                        callback=[RLModelLoggingCallback(), checkpoint_callback]
                    )
                    self.model.save(model_path)
                    # print(f'{interval_index % len(interval_list)}/{len(interval_list)} finished. total volume: {self.model.env.data_handler.total_volume}')
                    phase_timesteps -= interval
                    phase['phase_timesteps'] = phase_timesteps
                    config['current_stage'] = phase_index
                    config['phases'][phase_index] = phase
                    self.save_config(config, config_path)
                except Exception as e:
                    print(f'exception occurred: {e}: {traceback.format_exc()}')
                    self.model.save(model_path)
                    phase['phase_timesteps'] = phase_timesteps
                    config['current_stage'] = phase_index
                    config['phases'][phase_index] = phase
                    self.save_config(config, config_path)
                    continue

        print(f'Training completed over total timesteps: {total_timesteps}')
