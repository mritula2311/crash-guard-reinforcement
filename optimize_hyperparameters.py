#!/usr/bin/env python3
"""
Hyperparameter Optimization for CrashGuard Model

This script uses Optuna to automatically find the best hyperparameters
for maximum accuracy improvement.
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crash_guard import CrashGuardEnv, CrashGuardEvaluator
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn


def create_optimized_env(balanced_severity=True, enhanced_rewards=True):
    """Create optimized environment for hyperparameter tuning."""
    env_params = {
        'dataset_size': 15000,
        'test_mode': False,
        'random_seed': 42,
        'balanced_severity': balanced_severity,
        'enhanced_rewards': enhanced_rewards,
        'curriculum_learning': False  # Disable for consistent evaluation
    }
    
    env = CrashGuardEnv(**env_params)
    return env


def objective_dqn(trial):
    """Objective function for DQN hyperparameter optimization."""
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    buffer_size = trial.suggest_categorical('buffer_size', [50000, 100000, 200000])
    target_update_interval = trial.suggest_categorical('target_update_interval', [1000, 2000, 5000])
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1)
    gamma = trial.suggest_float('gamma', 0.98, 0.999)
    tau = trial.suggest_float('tau', 0.001, 0.01)
    
    # Network architecture
    layer1_size = trial.suggest_categorical('layer1_size', [128, 256, 512])
    layer2_size = trial.suggest_categorical('layer2_size', [64, 128, 256])
    layer3_size = trial.suggest_categorical('layer3_size', [32, 64, 128])
    
    policy_kwargs = {
        'net_arch': [layer1_size, layer2_size, layer3_size],
        'activation_fn': nn.ReLU
    }
    
    # Create environment
    env = create_optimized_env()
    env = Monitor(env, filename=None)
    
    # Create evaluation environment
    eval_env = create_optimized_env()
    eval_env = Monitor(eval_env, filename=None)
    
    # Create model with suggested hyperparameters
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        gamma=gamma,
        tau=tau,
        policy_kwargs=policy_kwargs,
        verbose=0
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=2500,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
        verbose=0
    )
    
    # Train model
    model.learn(total_timesteps=25000, callback=eval_callback)
    
    # Evaluate final performance
    evaluator = CrashGuardEvaluator(model=model)
    results = evaluator.evaluate_model(
        test_episodes=200,
        env_params={'dataset_size': 5000, 'test_mode': True, 'random_seed': 123},
        verbose=False
    )
    
    # Return accuracy as the objective to maximize
    return results['accuracy']


def objective_ppo(trial):
    """Objective function for PPO hyperparameter optimization."""
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    n_epochs = trial.suggest_categorical('n_epochs', [5, 10, 15, 20])
    gamma = trial.suggest_float('gamma', 0.98, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.01)
    vf_coef = trial.suggest_float('vf_coef', 0.25, 1.0)
    
    # Network architecture
    layer1_size = trial.suggest_categorical('layer1_size', [128, 256, 512])
    layer2_size = trial.suggest_categorical('layer2_size', [64, 128, 256])
    layer3_size = trial.suggest_categorical('layer3_size', [32, 64, 128])
    
    policy_kwargs = {
        'net_arch': {
            'pi': [layer1_size, layer2_size, layer3_size],
            'vf': [layer1_size, layer2_size, layer3_size]
        },
        'activation_fn': nn.ReLU
    }
    
    # Create environment
    env = create_optimized_env()
    env = Monitor(env, filename=None)
    
    # Create evaluation environment
    eval_env = create_optimized_env()
    eval_env = Monitor(eval_env, filename=None)
    
    # Create model with suggested hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        policy_kwargs=policy_kwargs,
        verbose=0
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=2500,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
        verbose=0
    )
    
    # Train model
    model.learn(total_timesteps=25000, callback=eval_callback)
    
    # Evaluate final performance
    evaluator = CrashGuardEvaluator(model=model)
    results = evaluator.evaluate_model(
        test_episodes=200,
        env_params={'dataset_size': 5000, 'test_mode': True, 'random_seed': 123},
        verbose=False
    )
    
    # Return accuracy as the objective to maximize
    return results['accuracy']


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for CrashGuard')
    
    parser.add_argument('--model', type=str, default='DQN',
                       choices=['DQN', 'PPO'],
                       help='Type of RL model to optimize')
    
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials')
    
    parser.add_argument('--study-name', type=str, default=None,
                       help='Name for the optimization study')
    
    parser.add_argument('--db-url', type=str, default=None,
                       help='Database URL for study persistence (optional)')
    
    parser.add_argument('--output-dir', type=str, default='optimization_results',
                       help='Directory to save optimization results')
    
    return parser.parse_args()


def main():
    """Main hyperparameter optimization function."""
    args = parse_arguments()
    
    print("="*70)
    print("CRASHGUARD HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Model Type: {args.model}")
    print(f"Number of Trials: {args.n_trials}")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"optimization_{args.model.lower()}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create study name
    study_name = args.study_name or f"crashguard_{args.model.lower()}_optimization_{timestamp}"
    
    # Create study
    if args.db_url:
        study = optuna.create_study(
            study_name=study_name,
            storage=args.db_url,
            direction='maximize',
            sampler=TPESampler(),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
    else:
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
    
    # Select objective function
    objective = objective_dqn if args.model == 'DQN' else objective_ppo
    
    # Run optimization
    print(f"\\nStarting hyperparameter optimization...")
    print(f"This will run {args.n_trials} trials to find the best parameters.")
    print("Each trial trains and evaluates a model - this may take a while...")
    
    study.optimize(objective, n_trials=args.n_trials)
    
    # Print results
    print(f"\\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_trial.value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_file = os.path.join(output_dir, 'optimization_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"CrashGuard {args.model} Hyperparameter Optimization Results\\n")
        f.write(f"Timestamp: {timestamp}\\n")
        f.write(f"Number of trials: {args.n_trials}\\n\\n")
        f.write(f"Best trial: {study.best_trial.number}\\n")
        f.write(f"Best accuracy: {study.best_trial.value:.4f}\\n\\n")
        f.write("Best parameters:\\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\\n")
    
    # Create training script with optimized parameters
    script_content = f'''#!/usr/bin/env python3
"""
Auto-generated training script with optimized hyperparameters
Generated on: {timestamp}
Best accuracy achieved: {study.best_trial.value:.4f}
"""

from crash_guard import CrashGuardTrainer

# Optimized hyperparameters
env_params = {{
    'dataset_size': 20000,
    'balanced_severity': True,
    'enhanced_rewards': True,
    'curriculum_learning': True
}}

model_params = {study.best_trial.params}

# Create and train model
trainer = CrashGuardTrainer(
    env_params=env_params,
    model_type='{args.model}',
    model_params=model_params
)

trainer.create_environment()
trainer.create_model()
model_path = trainer.train(total_timesteps=200000)

print(f"Optimized model saved to: {{model_path}}")
'''
    
    script_path = os.path.join(output_dir, f'train_optimized_{args.model.lower()}.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\\nResults saved to: {output_dir}")
    print(f"Optimized training script: {script_path}")
    print(f"\\nTo train with optimized parameters, run:")
    print(f"python {script_path}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
