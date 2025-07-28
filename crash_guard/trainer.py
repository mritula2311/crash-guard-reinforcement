import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
from .environment import CrashGuardEnv


class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress and metrics.
    """
    
    def __init__(self, log_dir: str, verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.locals.get('dones', [])) > 0 and any(self.locals['dones']):
            if len(self.locals.get('infos', [])) > 0:
                for info in self.locals['infos']:
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])
        
        return True
    
    def _on_training_end(self) -> None:
        # Save training statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        
        np.save(os.path.join(self.log_dir, 'training_stats.npy'), stats)


class CrashGuardTrainer:
    """
    Trainer for CrashGuard RL models using Stable-Baselines3.
    """
    
    def __init__(self, 
                 env_params: Dict = None,
                 model_type: str = 'DQN',
                 model_params: Dict = None,
                 log_dir: str = 'logs'):
        """
        Initialize the trainer.
        
        Args:
            env_params: Parameters for environment creation
            model_type: Type of RL model ('DQN' or 'PPO')
            model_params: Parameters for the RL model
            log_dir: Directory for logging and saving models
        """
        self.env_params = env_params or {}
        self.model_type = model_type
        self.model_params = model_params or {}
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize environment
        self.env = None
        self.model = None
        self.training_history = {}
        
    def create_environment(self) -> CrashGuardEnv:
        """Create and validate the training environment."""
        env = CrashGuardEnv(**self.env_params)
        
        # Check environment compatibility
        check_env(env)
        
        # Wrap with Monitor for logging
        env = Monitor(env, self.log_dir)
        
        self.env = env
        return env
    
    def create_model(self, env: CrashGuardEnv = None):
        """
        Create the RL model.
        
        Args:
            env: Environment to train on (uses self.env if None)
        """
        if env is None:
            env = self.env
        
        if env is None:
            raise ValueError("Environment must be created before model")
        
        # Default model parameters
        default_params = {
            'DQN': {
                'learning_rate': 1e-3,
                'buffer_size': 50000,
                'learning_starts': 1000,
                'batch_size': 32,
                'tau': 1.0,
                'gamma': 0.99,
                'train_freq': 4,
                'gradient_steps': 1,
                'target_update_interval': 1000,
                'exploration_fraction': 0.3,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                'tensorboard_log': self.log_dir
            },
            'PPO': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'clip_range_vf': None,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'tensorboard_log': self.log_dir
            }
        }
        
        # Merge with user parameters
        params = default_params[self.model_type].copy()
        params.update(self.model_params)
        
        # Create model
        if self.model_type == 'DQN':
            self.model = DQN('MlpPolicy', env, verbose=1, **params)
        elif self.model_type == 'PPO':
            self.model = PPO('MlpPolicy', env, verbose=1, **params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return self.model
    
    def train(self, 
              total_timesteps: int = 100000,
              save_path: str = None,
              callback_params: Dict = None) -> None:
        """
        Train the RL model.
        
        Args:
            total_timesteps: Number of timesteps to train
            save_path: Path to save the trained model
            callback_params: Parameters for training callback
        """
        if self.model is None:
            raise ValueError("Model must be created before training")
        
        # Create callback
        callback_params = callback_params or {}
        callback = TrainingCallback(self.log_dir, **callback_params)
        
        # Train the model
        print(f"Starting training with {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False  # Disable progress bar to avoid dependency issues
        )
        
        # Save model
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.log_dir, 
                f"crash_guard_{self.model_type.lower()}_{timestamp}"
            )
        
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")
        
        # Store training history
        self.training_history = {
            'total_timesteps': total_timesteps,
            'model_type': self.model_type,
            'save_path': save_path,
            'episode_rewards': callback.episode_rewards,
            'episode_lengths': callback.episode_lengths
        }
        
        return save_path
    
    def plot_training_progress(self, save_path: str = None) -> None:
        """
        Plot training progress and learning curves.
        
        Args:
            save_path: Path to save the plots
        """
        if not self.training_history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CrashGuard Training Progress', fontsize=16)
        
        episode_rewards = self.training_history['episode_rewards']
        episode_lengths = self.training_history['episode_lengths']
        
        if len(episode_rewards) == 0:
            print("No episode data available for plotting")
            return
        
        # Plot 1: Episode Rewards
        axes[0, 0].plot(episode_rewards, alpha=0.6, color='blue')
        if len(episode_rewards) > 100:
            # Moving average
            window = min(100, len(episode_rewards) // 10)
            rewards_smooth = pd.Series(episode_rewards).rolling(window).mean()
            axes[0, 0].plot(rewards_smooth, color='red', linewidth=2, label=f'MA({window})')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        axes[0, 1].plot(episode_lengths, alpha=0.6, color='green')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Reward Distribution
        axes[1, 0].hist(episode_rewards, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Learning Progress (Cumulative Reward)
        if len(episode_rewards) > 1:
            cumulative_rewards = np.cumsum(episode_rewards)
            axes[1, 1].plot(cumulative_rewards, color='orange', linewidth=2)
            axes[1, 1].set_title('Cumulative Rewards')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Cumulative Reward')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to: {save_path}")
        
        plt.show()
    
    def save_training_data(self, path: str = None) -> str:
        """
        Save training data and statistics.
        
        Args:
            path: Path to save the data
            
        Returns:
            Path where data was saved
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.log_dir, f"training_data_{timestamp}.npz")
        
        # Prepare data
        data = {
            'training_history': self.training_history,
            'env_params': self.env_params,
            'model_params': self.model_params,
            'model_type': self.model_type
        }
        
        np.savez(path, **data)
        print(f"Training data saved to: {path}")
        
        return path
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the saved model
        """
        if self.model_type == 'DQN':
            self.model = DQN.load(model_path)
        elif self.model_type == 'PPO':
            self.model = PPO.load(model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        print(f"Model loaded from: {model_path}")
        return self.model
    
    def quick_train(self, 
                   dataset_size: int = 5000,
                   total_timesteps: int = 50000,
                   model_type: str = 'DQN') -> str:
        """
        Quick training pipeline with default parameters.
        
        Args:
            dataset_size: Size of training dataset
            total_timesteps: Number of training timesteps
            model_type: Type of model to train
            
        Returns:
            Path to saved model
        """
        self.model_type = model_type
        self.env_params = {'dataset_size': dataset_size, 'random_seed': 42}
        
        # Create environment and model
        self.create_environment()
        self.create_model()
        
        # Train
        model_path = self.train(total_timesteps)
        
        # Generate plots
        plot_path = os.path.join(self.log_dir, 'training_progress.png')
        self.plot_training_progress(plot_path)
        
        # Save training data
        self.save_training_data()
        
        return model_path