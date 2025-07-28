"""
Example script demonstrating basic usage of the CrashGuard RL environment.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crash_guard import CrashGuardEnv, CrashDataGenerator


def demonstrate_environment():
    """Demonstrate basic environment functionality."""
    print("="*60)
    print("CRASHGUARD ENVIRONMENT DEMONSTRATION")
    print("="*60)
    
    # Create environment
    print("\n1. Creating CrashGuard environment...")
    env = CrashGuardEnv(dataset_size=100, random_seed=42)
    
    print(f"   Observation Space: {env.observation_space}")
    print(f"   Action Space: {env.action_space}")
    print(f"   Feature Names: {env.feature_names}")
    
    # Show data generator descriptions
    print("\n2. Feature Descriptions:")
    descriptions = env.data_generator.get_feature_descriptions()
    for feature, desc in descriptions.items():
        print(f"   {feature}: {desc}")
    
    print("\n3. Action Descriptions:")
    actions = env.data_generator.get_action_descriptions()
    for action_id, desc in actions.items():
        print(f"   {action_id}: {desc}")
    
    # Run some example episodes
    print("\n4. Running example episodes...")
    
    for episode in range(5):
        obs = env.reset()
        
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Initial State: {obs}")
        
        # Show rendered environment
        env.render()
        
        # Take random action
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        
        print(f"Action Taken: {action} ({actions[action]})")
        print(f"Reward: {reward}")
        print(f"Episode Info: {info}")
    
    # Show environment statistics
    print("\n5. Environment Statistics:")
    stats = env.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    env.close()


def demonstrate_data_generator():
    """Demonstrate data generation capabilities."""
    print("\n\n" + "="*60)
    print("DATA GENERATOR DEMONSTRATION")
    print("="*60)
    
    # Create data generator
    data_gen = CrashDataGenerator(random_seed=42)
    
    # Generate individual scenarios
    print("\n1. Generating individual crash scenarios...")
    
    for severity in ['minor', 'moderate', 'severe']:
        scenario = data_gen.generate_crash_scenario(severity)
        print(f"\n{severity.upper()} Crash Scenario:")
        for key, value in scenario.items():
            if key not in ['true_severity', 'severity_label']:
                print(f"   {key}: {value:.3f}")
        print(f"   Severity: {scenario['severity_label']} (Level {scenario['true_severity']})")
    
    # Generate full dataset
    print("\n2. Generating dataset...")
    dataset = data_gen.generate_dataset(n_samples=1000)
    
    print(f"   Dataset shape: {dataset.shape}")
    print(f"   Columns: {list(dataset.columns)}")
    
    # Show severity distribution
    severity_dist = dataset['severity_label'].value_counts()
    print(f"\n   Severity Distribution:")
    for severity, count in severity_dist.items():
        print(f"     {severity}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Show feature statistics
    print(f"\n3. Feature Statistics:")
    print(dataset.describe())
    
    # Normalize and show normalized stats
    normalized_dataset = data_gen.normalize_features(dataset)
    print(f"\n4. Normalized Feature Statistics:")
    feature_cols = [col for col in normalized_dataset.columns 
                   if col not in ['true_severity', 'severity_label']]
    print(normalized_dataset[feature_cols].describe())


def demonstrate_reward_function():
    """Demonstrate the reward function behavior."""
    print("\n\n" + "="*60)
    print("REWARD FUNCTION DEMONSTRATION")
    print("="*60)
    
    env = CrashGuardEnv(dataset_size=10, random_seed=42)
    
    # Test all actions for different severity levels
    action_names = [
        "High-Priority Alert",
        "Wait for Confirmation",
        "Local Safety Mechanism",
        "Broadcast to Vehicles", 
        "Low-Priority Notification"
    ]
    
    severity_names = ["Minor", "Moderate", "Severe"]
    
    print("\nReward Matrix (Action vs Severity):")
    print(f"{'Action':<25} {'Minor':<8} {'Moderate':<10} {'Severe':<8}")
    print("-" * 55)
    
    # Create test scenarios for each severity
    test_scenarios = []
    for severity_level in [0, 1, 2]:  # minor, moderate, severe
        obs = env.reset()
        # Force specific severity for testing
        env.current_scenario['true_severity'] = severity_level
        env.current_scenario['severity_label'] = severity_names[severity_level]
        test_scenarios.append(env.current_scenario.copy())
    
    # Test each action against each severity
    for action_id in range(5):
        rewards = []
        for scenario in test_scenarios:
            env.current_scenario = scenario
            reward = env._calculate_reward(action_id)
            rewards.append(reward)
        
        print(f"{action_names[action_id]:<25} {rewards[0]:<8.1f} {rewards[1]:<10.1f} {rewards[2]:<8.1f}")
    
    env.close()


if __name__ == "__main__":
    demonstrate_environment()
    demonstrate_data_generator()
    demonstrate_reward_function()