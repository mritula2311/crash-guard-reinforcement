#!/usr/bin/env python3
"""
Main training script for CrashGuard RL models.

Usage:
    python train.py --model DQN --timesteps 100000 --dataset-size 10000
    python train.py --model PPO --timesteps 50000 --dataset-size 5000
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crash_guard import CrashGuardTrainer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CrashGuard RL Model')
    
    parser.add_argument('--model', type=str, default='DQN', 
                       choices=['DQN', 'PPO'],
                       help='Type of RL model to train')
    
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of training timesteps')
    
    parser.add_argument('--dataset-size', type=int, default=10000,
                       help='Size of training dataset')
    
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for logging and saving models')
    
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate for the model')
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick training with default parameters')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    print("="*60)
    print("CRASHGUARD RL MODEL TRAINING")
    print("="*60)
    print(f"Model Type: {args.model}")
    print(f"Training Timesteps: {args.timesteps}")
    print(f"Dataset Size: {args.dataset_size}")
    print(f"Random Seed: {args.random_seed}")
    print(f"Log Directory: {args.log_dir}")
    
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"training_{args.model.lower()}_{timestamp}")
    
    try:
        if args.quick:
            print("\nRunning quick training...")
            trainer = CrashGuardTrainer(log_dir=log_dir)
            model_path = trainer.quick_train(
                dataset_size=args.dataset_size,
                total_timesteps=args.timesteps,
                model_type=args.model
            )
        else:
            # Setup environment parameters
            env_params = {
                'dataset_size': args.dataset_size,
                'random_seed': args.random_seed,
                'test_mode': False
            }
            
            # Setup model parameters
            model_params = {}
            if args.learning_rate is not None:
                model_params['learning_rate'] = args.learning_rate
            
            # Initialize trainer
            trainer = CrashGuardTrainer(
                env_params=env_params,
                model_type=args.model,
                model_params=model_params,
                log_dir=log_dir
            )
            
            # Create environment and model
            print("\nCreating environment...")
            trainer.create_environment()
            
            print("Creating model...")
            trainer.create_model()
            
            # Train the model
            print("\nStarting training...")
            model_path = trainer.train(total_timesteps=args.timesteps)
            
            # Generate training plots
            print("\nGenerating training plots...")
            plot_path = os.path.join(log_dir, 'training_progress.png')
            trainer.plot_training_progress(plot_path)
            
            # Save training data
            print("Saving training data...")
            trainer.save_training_data()
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Logs saved to: {log_dir}")
        
        # Quick evaluation
        print("\nRunning quick evaluation...")
        from crash_guard import CrashGuardEvaluator
        
        evaluator = CrashGuardEvaluator(model_path, args.model)
        results = evaluator.evaluate_model(
            test_episodes=500,
            env_params={'dataset_size': 500, 'test_mode': True, 'random_seed': 123}
        )
        
        # Save evaluation plots
        eval_plot_path = os.path.join(log_dir, 'evaluation_results.png')
        evaluator.plot_evaluation_results(eval_plot_path)
        
        print(f"\nEvaluation completed!")
        print(f"Mean Reward: {results['mean_reward']:.3f}")
        print(f"Accuracy: {results['accuracy']:.3f}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)