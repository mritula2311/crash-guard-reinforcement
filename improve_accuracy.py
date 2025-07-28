#!/usr/bin/env python3
"""
Quick Accuracy Improvement Script for CrashGuard

This script provides immediate improvements to address the current model's issues:
1. Model only using one action (Broadcast to Vehicles)
2. Poor accuracy on minor crashes (0%)
3. Overall accuracy of 60%
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crash_guard import CrashGuardTrainer, CrashGuardEvaluator


def train_improved_model():
    """Train an improved model with better accuracy."""
    print("="*60)
    print("QUICK ACCURACY IMPROVEMENT TRAINING")
    print("="*60)
    
    # Enhanced environment parameters
    env_params = {
        'dataset_size': 15000,  # Larger dataset for better diversity
        'random_seed': 42,
        'test_mode': False,
        'balanced_severity': True,  # Equal distribution of crash types
        'enhanced_rewards': True,   # Better reward shaping
    }
    
    # Enhanced DQN parameters to fix the action selection issue
    model_params = {
        'learning_rate': 3e-4,      # Lower learning rate for stability
        'batch_size': 64,           # Smaller batch for better gradient updates
        'buffer_size': 50000,
        'learning_starts': 2000,    # More learning before training starts
        'target_update_interval': 1000,
        'exploration_fraction': 0.5,  # Longer exploration phase
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.01,  # Lower final epsilon
        'gamma': 0.99,              # Standard discount factor
        'tau': 0.001,               # Slower target network updates
        'train_freq': 4,
        'gradient_steps': 1
    }
    
    print("Enhanced Parameters:")
    print(f"- Balanced training: {env_params['balanced_severity']}")
    print(f"- Enhanced rewards: {env_params['enhanced_rewards']}")
    print(f"- Learning rate: {model_params['learning_rate']}")
    print(f"- Exploration fraction: {model_params['exploration_fraction']}")
    print(f"- Final epsilon: {model_params['exploration_final_eps']}")
    
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"improved_training_dqn_{timestamp}")
    
    # Initialize trainer with enhanced parameters
    trainer = CrashGuardTrainer(
        env_params=env_params,
        model_type='DQN',
        model_params=model_params,
        log_dir=log_dir
    )
    
    # Create environment and model
    print("\\nCreating enhanced environment...")
    trainer.create_environment()
    
    print("Creating improved DQN model...")
    trainer.create_model()
    
    # Train the model with more timesteps
    print("\\nStarting improved training...")
    print("This training addresses the following issues:")
    print("- Action diversity (not just broadcasting)")
    print("- Better minor crash detection")
    print("- Improved overall accuracy")
    
    model_path = trainer.train(total_timesteps=100000)
    
    # Generate training plots
    print("\\nGenerating training plots...")
    plot_path = os.path.join(log_dir, 'training_progress.png')
    trainer.plot_training_progress(plot_path)
    
    # Save training data
    trainer.save_training_data()
    
    return model_path, log_dir


def evaluate_improvement(model_path):
    """Evaluate the improved model and compare with baseline."""
    print("\\n" + "="*60)
    print("EVALUATING IMPROVED MODEL")
    print("="*60)
    
    # Evaluate the new model
    evaluator = CrashGuardEvaluator(model_path, 'DQN')
    
    print("Running comprehensive evaluation...")
    results = evaluator.evaluate_model(
        test_episodes=500,
        env_params={
            'dataset_size': 2000, 
            'test_mode': True, 
            'random_seed': 123,
            'balanced_severity': True
        },
        verbose=True
    )
    
    print("\\n" + "="*60)
    print("IMPROVED MODEL RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    
    if 'severe_crash_response_rate' in results:
        print(f"Severe Crash Detection: {results['severe_crash_response_rate']:.3f} ({results['severe_crash_response_rate']*100:.1f}%)")
    
    if 'false_alarm_rate' in results:
        print(f"False Alarm Rate: {results['false_alarm_rate']:.3f} ({results['false_alarm_rate']*100:.1f}%)")
    
    # Action distribution analysis
    print("\\nAction Distribution (should be more diverse):")
    action_names = [
        "High-Priority Alert",
        "Wait for Confirmation", 
        "Local Safety Mechanism",
        "Broadcast to Vehicles",
        "Low-Priority Notification"
    ]
    
    for i, (name, prob) in enumerate(zip(action_names, results['action_distribution'])):
        print(f"  {name}: {prob:.3f} ({prob*100:.1f}%)")
    
    # Check if action diversity improved
    max_action_prob = max(results['action_distribution'])
    if max_action_prob < 0.8:
        print("\\n✅ IMPROVEMENT: Better action diversity achieved!")
    else:
        print(f"\\n⚠ Still concentrated on one action ({max_action_prob*100:.1f}%)")
    
    # Severity-specific performance
    print("\\nSeverity-Specific Performance:")
    severity_names = ["Minor", "Moderate", "Severe"]
    
    for i, name in enumerate(severity_names):
        count_key = f'severity_{i}_count'
        accuracy_key = f'severity_{i}_accuracy'
        
        if count_key in results and results[count_key] > 0:
            accuracy = results[accuracy_key]
            print(f"  {name} Crashes: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            if i == 0 and accuracy > 0.5:  # Minor crashes
                print("    ✅ IMPROVEMENT: Better minor crash handling!")
            elif i == 0 and accuracy <= 0.1:
                print("    ⚠ Still struggling with minor crashes")
    
    return results


def compare_with_baseline():
    """Compare with the original model performance."""
    print("\\n" + "="*60)
    print("COMPARISON WITH BASELINE")
    print("="*60)
    print("Original Model Performance:")
    print("- Overall Accuracy: 60.0%")
    print("- Action Distribution: 100% Broadcast to Vehicles")
    print("- Minor Crash Accuracy: 0.0%")
    print("- Issues: Poor action diversity, no minor crash detection")
    
    print("\\nImproved Model should show:")
    print("- Higher overall accuracy (>70%)")
    print("- Better action diversity (<80% single action)")
    print("- Improved minor crash detection (>30%)")
    print("- Maintained severe crash detection (>90%)")


def main():
    """Main function for quick accuracy improvement."""
    try:
        print("Starting quick accuracy improvement for CrashGuard model...")
        
        # Show baseline comparison
        compare_with_baseline()
        
        # Train improved model
        model_path, log_dir = train_improved_model()
        
        print(f"\\nImproved model training completed!")
        print(f"Model saved to: {model_path}")
        print(f"Logs saved to: {log_dir}")
        
        # Evaluate improvements
        results = evaluate_improvement(model_path)
        
        # Final recommendations
        print("\\n" + "="*60)
        print("NEXT STEPS FOR FURTHER IMPROVEMENT")
        print("="*60)
        
        if results['accuracy'] > 0.75:
            print("🎉 Great improvement! For even better results:")
            print("- Run hyperparameter optimization: python optimize_hyperparameters.py")
            print("- Try curriculum learning: add --curriculum-learning")
        elif results['accuracy'] > 0.65:
            print("👍 Good improvement! To reach higher accuracy:")
            print("- Increase training time: --timesteps 150000")
            print("- Try PPO model: --model PPO")
        else:
            print("📈 Some improvement, continue with:")
            print("- Longer training: --timesteps 200000")
            print("- Check environment parameters")
        
        print(f"\\nTo evaluate this model later:")
        print(f"python evaluate.py --model-path {model_path} --episodes 1000 --generate-heatmap")
        
        return 0
        
    except Exception as e:
        print(f"\\nError during improvement training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
