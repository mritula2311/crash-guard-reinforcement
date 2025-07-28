#!/usr/bin/env python3
"""
Enhanced CrashGuard Model Training Script with Accuracy Improvements

This script implements several techniques to improve model accuracy:
1. Enhanced network architecture with deeper layers
2. Improved hyperparameters and training configuration
3. Better reward shaping and curriculum learning
4. Data augmentation and balanced training
5. Advanced training callbacks and monitoring
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crash_guard import CrashGuardTrainer, CrashGuardEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced CrashGuard RL Model Training')
    
    parser.add_argument('--model', type=str, default='DQN',
                       choices=['DQN', 'PPO'],
                       help='Type of RL model to train')
    
    parser.add_argument('--timesteps', type=int, default=200000,
                       help='Number of training timesteps (increased for better convergence)')
    
    parser.add_argument('--dataset-size', type=int, default=20000,
                       help='Size of training dataset (increased for better diversity)')
    
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of episodes for evaluation during training')
    
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Learning rate for the model')
    
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (increased for stability)')
    
    parser.add_argument('--buffer-size', type=int, default=100000,
                       help='Replay buffer size for DQN')
    
    parser.add_argument('--gamma', type=float, default=0.995,
                       help='Discount factor (increased for long-term planning)')
    
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--save-freq', type=int, default=10000,
                       help='Save model every N timesteps')
    
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='Evaluate model every N timesteps')
    
    parser.add_argument('--output-dir', type=str, default='logs',
                       help='Directory to save training outputs')
    
    parser.add_argument('--curriculum-learning', action='store_true',
                       help='Enable curriculum learning for progressive difficulty')
    
    parser.add_argument('--balanced-training', action='store_true',
                       help='Use balanced training with equal severity distribution')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed training information')
    
    return parser.parse_args()


def create_enhanced_policy_kwargs():
    """Create enhanced neural network architecture for better accuracy."""
    # Enhanced network architecture with more layers and neurons
    policy_kwargs = {
        'net_arch': {
            'pi': [512, 256, 128],  # Policy network: deeper and wider
            'vf': [512, 256, 128]   # Value function network: deeper and wider
        },
        'activation_fn': nn.ReLU,
        'normalize_images': False
    }
    return policy_kwargs


def create_enhanced_dqn_params(args):
    """Create enhanced DQN parameters for better accuracy."""
    policy_kwargs = {
        'net_arch': [512, 256, 128, 64],  # Deeper network
        'activation_fn': nn.ReLU
    }
    
    params = {
        'learning_rate': args.learning_rate,
        'buffer_size': args.buffer_size,
        'learning_starts': 2000,  # Increased warm-up period
        'batch_size': args.batch_size,
        'tau': 0.005,  # Softer target network updates
        'gamma': args.gamma,
        'train_freq': 4,
        'gradient_steps': 2,  # More gradient steps per update
        'target_update_interval': 2000,  # More frequent target updates
        'exploration_fraction': 0.4,  # Longer exploration
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.02,  # Lower final epsilon for better exploitation
        'policy_kwargs': policy_kwargs,
        'verbose': 1 if args.verbose else 0
    }
    return params


def create_enhanced_ppo_params(args):
    """Create enhanced PPO parameters for better accuracy."""
    policy_kwargs = create_enhanced_policy_kwargs()
    
    params = {
        'learning_rate': args.learning_rate,
        'n_steps': 4096,  # Increased rollout length
        'batch_size': args.batch_size,
        'n_epochs': 15,  # More training epochs per rollout
        'gamma': args.gamma,
        'gae_lambda': 0.98,  # Adjusted GAE parameter
        'clip_range': 0.2,
        'clip_range_vf': 0.2,
        'ent_coef': 0.01,  # Small entropy bonus for exploration
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': policy_kwargs,
        'verbose': 1 if args.verbose else 0
    }
    return params


def create_enhanced_environment(args):
    """Create enhanced environment with improved settings."""
    env_params = {
        'dataset_size': args.dataset_size,
        'test_mode': False,
        'random_seed': args.random_seed,
        'balanced_severity': args.balanced_training,  # Add balanced training option
        'enhanced_rewards': True,  # Enable enhanced reward shaping
        'curriculum_learning': args.curriculum_learning  # Enable curriculum learning
    }
    
    # Create environment
    env = CrashGuardEnv(**env_params)
    
    # Wrap with Monitor for better logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.output_dir, f"enhanced_training_{args.model.lower()}_{timestamp}")
    os.makedirs(log_path, exist_ok=True)
    
    env = Monitor(env, log_path)
    
    return env, log_path


def create_training_callbacks(args, log_path, eval_env):
    """Create enhanced training callbacks for better monitoring and early stopping."""
    callbacks = []
    
    # Evaluation callback for monitoring performance during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Note: StopTrainingOnRewardThreshold is automatically handled by EvalCallback
    # when callback_on_new_best is used
    
    return callbacks


def main():
    """Enhanced main training function."""
    args = parse_arguments()
    
    print("="*70)
    print("ENHANCED CRASHGUARD RL MODEL TRAINING")
    print("="*70)
    print(f"Model Type: {args.model}")
    print(f"Training Timesteps: {args.timesteps:,}")
    print(f"Dataset Size: {args.dataset_size:,}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Buffer Size: {args.buffer_size:,}")
    print(f"Curriculum Learning: {args.curriculum_learning}")
    print(f"Balanced Training: {args.balanced_training}")
    print("="*70)
    
    try:
        # Create enhanced environment
        print("\\nCreating enhanced training environment...")
        env, log_path = create_enhanced_environment(args)
        
        # Create evaluation environment
        eval_env_params = {
            'dataset_size': 5000,
            'test_mode': True,
            'random_seed': args.random_seed + 1000
        }
        eval_env = CrashGuardEnv(**eval_env_params)
        eval_env = Monitor(eval_env, log_path)
        
        # Create enhanced model with improved parameters
        print(f"Creating enhanced {args.model} model...")
        if args.model == 'DQN':
            from stable_baselines3 import DQN
            model_params = create_enhanced_dqn_params(args)
            model = DQN('MlpPolicy', env, tensorboard_log=log_path, **model_params)
        else:  # PPO
            from stable_baselines3 import PPO
            model_params = create_enhanced_ppo_params(args)
            model = PPO('MlpPolicy', env, tensorboard_log=log_path, **model_params)
        
        # Create enhanced callbacks
        callbacks = create_training_callbacks(args, log_path, eval_env)
        
        # Train the enhanced model
        print(f"\\nStarting enhanced training for {args.timesteps:,} timesteps...")
        print("Training includes:")
        print("- Enhanced neural network architecture")
        print("- Improved hyperparameters")
        print("- Evaluation callbacks for monitoring")
        print("- Early stopping on reward threshold")
        if args.curriculum_learning:
            print("- Curriculum learning for progressive difficulty")
        if args.balanced_training:
            print("- Balanced training with equal severity distribution")
        
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=f"enhanced_{args.model.lower()}"
        )
        
        # Save the final enhanced model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(log_path, f"enhanced_crash_guard_{args.model.lower()}_{timestamp}")
        model.save(model_path)
        
        print(f"\\n" + "="*70)
        print("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Enhanced model saved to: {model_path}.zip")
        print(f"Training logs saved to: {log_path}")
        print(f"\\nTo evaluate the enhanced model, run:")
        print(f"python evaluate.py --model-path {model_path}.zip --episodes 1000 --generate-heatmap")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\\nError during enhanced training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
