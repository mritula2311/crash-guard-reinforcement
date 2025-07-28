#!/usr/bin/env python3
"""
Quick Model Evaluation Script

This script automatically finds and evaluates the best available models
in your logs directory, handling both DQN and PPO models correctly.
"""

import os
import sys
import glob
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crash_guard import CrashGuardEvaluator


def find_available_models():
    """Find all available trained models in the logs directory."""
    models = []
    
    # Search patterns for different model types
    search_patterns = [
        "logs/enhanced_training_*/best_model.zip",
        "logs/enhanced_training_*/enhanced_crash_guard_*.zip",
        "logs/training_*/crash_guard_*.zip",
        "logs/improved_*/improved_crash_guard_*.zip"
    ]
    
    for pattern in search_patterns:
        found_models = glob.glob(pattern)
        for model_path in found_models:
            # Detect model type from path
            if 'ppo' in model_path.lower():
                model_type = 'PPO'
            elif 'dqn' in model_path.lower():
                model_type = 'DQN'
            else:
                # Try to detect from directory structure
                model_type = 'DQN'  # Default
            
            # Get modification time for sorting
            mod_time = os.path.getmtime(model_path)
            
            models.append({
                'path': model_path,
                'type': model_type,
                'mod_time': mod_time,
                'size': os.path.getsize(model_path) / (1024*1024)  # MB
            })
    
    # Sort by modification time (newest first)
    models.sort(key=lambda x: x['mod_time'], reverse=True)
    
    return models


def evaluate_model(model_info, episodes=500, verbose=True):
    """Evaluate a single model."""
    model_path = model_info['path']
    model_type = model_info['type']
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {os.path.basename(model_path)}")
    print(f"Type: {model_type} | Size: {model_info['size']:.1f} MB")
    print(f"{'='*60}")
    
    try:
        # Create evaluator
        evaluator = CrashGuardEvaluator(model_path, model_type)
        
        # Run evaluation with enhanced environment
        print(f"Running evaluation with {episodes} episodes...")
        results = evaluator.evaluate_model(
            test_episodes=episodes,
            env_params={
                'dataset_size': min(episodes * 2, 2000), 
                'test_mode': True, 
                'random_seed': 123,
                'balanced_severity': True,  # Use balanced test set
                'enhanced_rewards': False   # Use standard rewards for fair comparison
            },
            verbose=verbose
        )
        
        # Print key results
        print(f"\n🎯 PERFORMANCE RESULTS:")
        print(f"Overall Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        
        if 'severe_crash_response_rate' in results:
            rate = results['severe_crash_response_rate']
            print(f"Severe Crash Detection: {rate:.3f} ({rate*100:.1f}%)")
        
        if 'false_alarm_rate' in results:
            print(f"False Alarm Rate: {results['false_alarm_rate']:.3f} ({results['false_alarm_rate']*100:.1f}%)")
        
        # Action distribution analysis
        action_names = [
            "High-Priority Alert", "Wait for Confirmation", "Local Safety",
            "Broadcast to Vehicles", "Low-Priority Notification"
        ]
        
        print(f"\n📊 ACTION DISTRIBUTION:")
        max_action_prob = max(results['action_distribution'])
        for i, (name, prob) in enumerate(zip(action_names, results['action_distribution'])):
            marker = "⭐" if prob == max_action_prob else "  "
            print(f"{marker} {name}: {prob:.3f} ({prob*100:.1f}%)")
        
        # Check for issues
        if max_action_prob > 0.8:
            print(f"\n⚠️  WARNING: High concentration on single action ({max_action_prob*100:.1f}%)")
            print("   Consider training with balanced data or enhanced rewards")
        
        if results['accuracy'] < 0.7:
            print(f"\n📈 IMPROVEMENT SUGGESTIONS:")
            print("   - Run: python improve_accuracy.py")
            print("   - Try: python train_enhanced.py --balanced-training")
        elif results['accuracy'] > 0.85:
            print(f"\n🎉 EXCELLENT PERFORMANCE! This model is ready for deployment.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error evaluating model: {e}")
        return None


def compare_models(model_results):
    """Compare multiple model results."""
    if len(model_results) < 2:
        return
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    
    # Sort by accuracy
    sorted_results = sorted(model_results, key=lambda x: x[1]['accuracy'] if x[1] else 0, reverse=True)
    
    print(f"{'Rank':<4} {'Model':<40} {'Type':<4} {'Accuracy':<10} {'Reward':<10}")
    print(f"{'-'*80}")
    
    for i, (model_info, results) in enumerate(sorted_results):
        if results:
            model_name = os.path.basename(model_info['path'])[:35] + "..."
            accuracy = f"{results['accuracy']*100:.1f}%"
            reward = f"{results['mean_reward']:.2f}"
            print(f"{i+1:<4} {model_name:<40} {model_info['type']:<4} {accuracy:<10} {reward:<10}")
    
    # Best model recommendation
    if sorted_results[0][1]:
        best_model = sorted_results[0][0]
        best_results = sorted_results[0][1]
        print(f"\n🏆 BEST MODEL: {os.path.basename(best_model['path'])}")
        print(f"   Accuracy: {best_results['accuracy']*100:.1f}%")
        print(f"   Use this for deployment!")


def main():
    """Main evaluation function."""
    print("🔍 AUTOMATIC MODEL EVALUATION")
    print("="*60)
    print("Searching for trained models...")
    
    # Find available models
    models = find_available_models()
    
    if not models:
        print("❌ No trained models found!")
        print("\nTo train a model, run:")
        print("python train_enhanced.py --model DQN --balanced-training")
        print("python train_enhanced.py --model PPO --balanced-training")
        return 1
    
    print(f"✅ Found {len(models)} trained models")
    
    # Show available models
    print(f"\nAvailable models:")
    for i, model in enumerate(models):
        mod_time = datetime.fromtimestamp(model['mod_time']).strftime('%Y-%m-%d %H:%M')
        print(f"  {i+1}. {os.path.basename(model['path'])} ({model['type']}) - {mod_time}")
    
    # Evaluate models
    model_results = []
    
    # Evaluate up to 3 most recent models
    models_to_evaluate = models[:3]
    
    for model_info in models_to_evaluate:
        results = evaluate_model(model_info, episodes=300, verbose=False)
        model_results.append((model_info, results))
    
    # Compare results
    compare_models(model_results)
    
    print(f"\n{'='*60}")
    print("🚀 NEXT STEPS:")
    print("="*60)
    
    if any(results[1] and results[1]['accuracy'] > 0.85 for results in model_results):
        print("✅ You have high-accuracy models ready for deployment!")
    else:
        print("📈 To improve accuracy further:")
        print("   1. python improve_accuracy.py")
        print("   2. python optimize_hyperparameters.py --model DQN")
        print("   3. python train_enhanced.py --curriculum-learning --balanced-training")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
