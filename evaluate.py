#!/usr/bin/env python3
"""
Main evaluation script for CrashGuard RL models.

Usage:
    python evaluate.py --model-path logs/crash_guard_dqn_20231201_120000.zip --model-type DQN
    python evaluate.py --model-path logs/crash_guard_ppo_20231201_120000.zip --episodes 1000
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crash_guard import CrashGuardEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate CrashGuard RL Model')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model file')
    
    parser.add_argument('--model-type', type=str, default='DQN',
                       choices=['DQN', 'PPO'],
                       help='Type of RL model')
    
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of test episodes')
    
    parser.add_argument('--dataset-size', type=int, default=None,
                       help='Size of test dataset (defaults to episodes)')
    
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    parser.add_argument('--random-seed', type=int, default=123,
                       help='Random seed for test environment')
    
    parser.add_argument('--generate-heatmap', action='store_true',
                       help='Generate policy behavior heatmap')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    print("="*60)
    print("CRASHGUARD RL MODEL EVALUATION")
    print("="*60)
    print(f"Model Path: {args.model_path}")
    print(f"Model Type: {args.model_type}")
    print(f"Test Episodes: {args.episodes}")
    print(f"Random Seed: {args.random_seed}")
    
    # Verify model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize evaluator
        print("\nLoading model...")
        evaluator = CrashGuardEvaluator(args.model_path, args.model_type)
        
        # Setup test environment parameters
        dataset_size = args.dataset_size or args.episodes
        env_params = {
            'dataset_size': dataset_size,
            'test_mode': True,
            'random_seed': args.random_seed
        }
        
        # Run evaluation
        print(f"\nRunning evaluation on {args.episodes} episodes...")
        results = evaluator.evaluate_model(
            test_episodes=args.episodes,
            env_params=env_params,
            verbose=args.verbose
        )
        
        # Generate evaluation plots
        print("\nGenerating evaluation plots...")
        plot_path = os.path.join(output_dir, 'evaluation_results.png')
        evaluator.plot_evaluation_results(plot_path)
        
        # Generate policy heatmap if requested
        if args.generate_heatmap:
            print("Generating policy heatmap...")
            heatmap_path = os.path.join(output_dir, 'policy_heatmap.png')
            evaluator.generate_policy_heatmap(env_params, heatmap_path)
        
        # Save evaluation results
        results_path = os.path.join(output_dir, 'evaluation_results.npz')
        evaluator.save_evaluation_results(results_path)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Print key metrics
        print(f"\n" + "="*40)
        print("KEY PERFORMANCE METRICS")
        print("="*40)
        print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"Total Reward: {results['total_reward']:.1f}")
        print(f"Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        
        if 'severe_crash_response_rate' in results:
            rate = results['severe_crash_response_rate']
            print(f"Severe Crash Response Rate: {rate:.3f} ({rate*100:.1f}%)")
        
        if 'false_alarm_rate' in results:
            print(f"False Alarm Rate: {results['false_alarm_rate']:.3f}")
        
        if 'missed_severe_rate' in results:
            print(f"Missed Severe Crash Rate: {results['missed_severe_rate']:.3f}")
        
        # Action distribution
        print(f"\nAction Distribution:")
        action_names = [
            "High-Priority Alert",
            "Wait for Confirmation",
            "Local Safety Mechanism", 
            "Broadcast to Vehicles",
            "Low-Priority Notification"
        ]
        
        for i, (name, prob) in enumerate(zip(action_names, results['action_distribution'])):
            print(f"  {name}: {prob:.3f} ({prob*100:.1f}%)")
        
        # Severity-specific performance
        print(f"\nSeverity-Specific Performance:")
        severity_names = ["Minor", "Moderate", "Severe"]
        for i, name in enumerate(severity_names):
            count_key = f'severity_{i}_count'
            reward_key = f'severity_{i}_mean_reward'
            accuracy_key = f'severity_{i}_accuracy'
            
            if count_key in results and results[count_key] > 0:
                print(f"  {name} Crashes ({results[count_key]} episodes):")
                print(f"    Mean Reward: {results[reward_key]:.3f}")
                print(f"    Accuracy: {results[accuracy_key]:.3f} ({results[accuracy_key]*100:.1f}%)")
        
        print("="*40)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)