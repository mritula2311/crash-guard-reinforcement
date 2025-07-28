import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from stable_baselines3 import DQN, PPO
from sklearn.metrics import confusion_matrix, classification_report
import os
from datetime import datetime
from .environment import CrashGuardEnv
from .data_generator import CrashDataGenerator


class CrashGuardEvaluator:
    """
    Evaluator for trained CrashGuard RL models.
    """
    
    def __init__(self, model_path: str = None, model_type: str = 'DQN'):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained model
            model_type: Type of RL model ('DQN' or 'PPO')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.evaluation_results = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            if self.model_type == 'DQN':
                self.model = DQN.load(model_path)
            elif self.model_type == 'PPO':
                self.model = PPO.load(model_path)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            print(f"Model loaded successfully from: {model_path}")
            self.model_path = model_path
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def evaluate_model(self, 
                      test_episodes: int = 1000,
                      env_params: Dict = None,
                      verbose: bool = True) -> Dict:
        """
        Evaluate the model on test episodes.
        
        Args:
            test_episodes: Number of test episodes
            env_params: Parameters for test environment
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None:
            raise ValueError("No model loaded for evaluation")
        
        # Create test environment
        env_params = env_params or {'dataset_size': test_episodes, 'test_mode': True}
        test_env = CrashGuardEnv(**env_params)
        
        # Storage for results
        episode_rewards = []
        episode_actions = []
        episode_severities = []
        episode_correct = []
        detailed_results = []
        
        # Run test episodes
        for episode in range(test_episodes):
            obs, info = test_env.reset()
            action, _states = self.model.predict(obs, deterministic=True)
            
            next_obs, reward, terminated, truncated, info = test_env.step(action)
            
            # Store results
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_severities.append(info['true_severity'])
            
            # Determine if action was appropriate
            is_correct = self._is_action_appropriate(action, info['true_severity'])
            episode_correct.append(is_correct)
            
            # Store detailed info
            detailed_results.append({
                'episode': episode,
                'action': action,
                'true_severity': info['true_severity'],
                'severity_label': info['severity_label'],
                'reward': reward,
                'correct': is_correct,
                'vehicle_speed': info['vehicle_speed'],
                'impact_g_force': info['impact_g_force'],
                'driver_drowsiness': info['driver_drowsiness']
            })
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Evaluated {episode + 1}/{test_episodes} episodes")
        
        # Calculate metrics
        results = self._calculate_metrics(
            episode_rewards, episode_actions, episode_severities, 
            episode_correct, detailed_results
        )
        
        # Get environment statistics
        env_stats = test_env.get_statistics()
        results.update(env_stats)
        
        self.evaluation_results = results
        
        if verbose:
            self.print_evaluation_summary()
        
        return results
    
    def _is_action_appropriate(self, action: int, true_severity: int) -> bool:
        """
        Determine if an action is appropriate for the given severity level.
        
        Args:
            action: Action taken (0-4)
            true_severity: True crash severity (0-2)
            
        Returns:
            Boolean indicating if action was appropriate
        """
        # Define appropriate actions for each severity level
        appropriate_actions = {
            0: [1, 2, 4],    # Minor: Wait, Local Safety, Low Priority
            1: [2, 3, 4],    # Moderate: Local Safety, Broadcast, Low Priority  
            2: [0, 3]        # Severe: High Priority, Broadcast
        }
        
        return action in appropriate_actions[true_severity]
    
    def _calculate_metrics(self, 
                          rewards: List[float],
                          actions: List[int], 
                          severities: List[int],
                          correct: List[bool],
                          detailed: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        
        metrics = {
            'total_episodes': len(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'total_reward': np.sum(rewards),
            'accuracy': np.mean(correct),
            'action_distribution': np.bincount(actions, minlength=5) / len(actions)
        }
        
        # Severity-specific metrics
        for severity in [0, 1, 2]:
            severity_mask = np.array(severities) == severity
            if np.any(severity_mask):
                severity_rewards = np.array(rewards)[severity_mask]
                severity_correct = np.array(correct)[severity_mask]
                severity_actions = np.array(actions)[severity_mask]
                
                metrics[f'severity_{severity}_count'] = np.sum(severity_mask)
                metrics[f'severity_{severity}_mean_reward'] = np.mean(severity_rewards)
                metrics[f'severity_{severity}_accuracy'] = np.mean(severity_correct)
                metrics[f'severity_{severity}_actions'] = np.bincount(severity_actions, minlength=5)
        
        # Action effectiveness
        for action in range(5):
            action_mask = np.array(actions) == action
            if np.any(action_mask):
                action_rewards = np.array(rewards)[action_mask]
                metrics[f'action_{action}_count'] = np.sum(action_mask)
                metrics[f'action_{action}_mean_reward'] = np.mean(action_rewards)
        
        # Safety metrics
        severe_crashes = np.array(severities) == 2
        if np.any(severe_crashes):
            severe_actions = np.array(actions)[severe_crashes]
            # High-priority alerts or broadcasts for severe crashes
            appropriate_severe_responses = np.isin(severe_actions, [0, 3])
            metrics['severe_crash_response_rate'] = np.mean(appropriate_severe_responses)
        
        return metrics
    
    def print_evaluation_summary(self):
        """Print a summary of evaluation results."""
        if not self.evaluation_results:
            print("No evaluation results available")
            return
        
        results = self.evaluation_results
        
        print("\n" + "="*60)
        print("CRASHGUARD MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Total Episodes: {results['total_episodes']}")
        print(f"  Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"  Total Reward: {results['total_reward']:.1f}")
        print(f"  Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        
        print(f"\nAction Distribution:")
        action_names = [
            "High-Priority Alert",
            "Wait for Confirmation", 
            "Local Safety Mechanism",
            "Broadcast to Vehicles",
            "Low-Priority Notification"
        ]
        
        for i, (name, prob) in enumerate(zip(action_names, results['action_distribution'])):
            print(f"  {i}: {name}: {prob:.3f} ({prob*100:.1f}%)")
        
        print(f"\nSeverity-Specific Performance:")
        severity_names = ["Minor", "Moderate", "Severe"]
        for i, name in enumerate(severity_names):
            count_key = f'severity_{i}_count'
            reward_key = f'severity_{i}_mean_reward'
            accuracy_key = f'severity_{i}_accuracy'
            
            if count_key in results:
                print(f"  {name} Crashes:")
                print(f"    Count: {results[count_key]}")
                print(f"    Mean Reward: {results[reward_key]:.3f}")
                print(f"    Accuracy: {results[accuracy_key]:.3f} ({results[accuracy_key]*100:.1f}%)")
        
        print(f"\nSafety Metrics:")
        if 'severe_crash_response_rate' in results:
            rate = results['severe_crash_response_rate']
            print(f"  Severe Crash Appropriate Response Rate: {rate:.3f} ({rate*100:.1f}%)")
        
        if 'false_alarm_rate' in results:
            print(f"  False Alarm Rate: {results['false_alarm_rate']:.3f}")
        
        if 'missed_severe_rate' in results:
            print(f"  Missed Severe Crash Rate: {results['missed_severe_rate']:.3f}")
        
        print("="*60)
    
    def plot_evaluation_results(self, save_path: str = None):
        """
        Create comprehensive evaluation plots.
        
        Args:
            save_path: Path to save the plots
        """
        if not self.evaluation_results:
            print("No evaluation results available for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CrashGuard Model Evaluation Results', fontsize=16)
        
        # Plot 1: Action Distribution
        action_names = ['High-Priority\nAlert', 'Wait for\nConfirmation', 
                       'Local Safety\nMechanism', 'Broadcast to\nVehicles', 
                       'Low-Priority\nNotification']
        action_dist = self.evaluation_results['action_distribution']
        
        axes[0, 0].bar(range(5), action_dist, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Action Distribution')
        axes[0, 0].set_xlabel('Action')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_xticks(range(5))
        axes[0, 0].set_xticklabels(action_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Severity Distribution
        severity_counts = []
        severity_names = ['Minor', 'Moderate', 'Severe']
        for i in range(3):
            count_key = f'severity_{i}_count'
            severity_counts.append(
                self.evaluation_results.get(count_key, 0)
            )
        
        axes[0, 1].pie(severity_counts, labels=severity_names, autopct='%1.1f%%',
                      colors=['lightgreen', 'orange', 'red'])
        axes[0, 1].set_title('Crash Severity Distribution')
        
        # Plot 3: Reward by Severity
        severity_rewards = []
        for i in range(3):
            reward_key = f'severity_{i}_mean_reward'
            severity_rewards.append(
                self.evaluation_results.get(reward_key, 0)
            )
        
        colors = ['green', 'orange', 'red']
        bars = axes[0, 2].bar(severity_names, severity_rewards, color=colors, alpha=0.7)
        axes[0, 2].set_title('Mean Reward by Severity')
        axes[0, 2].set_ylabel('Mean Reward')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, severity_rewards):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 4: Accuracy by Severity
        severity_accuracies = []
        for i in range(3):
            accuracy_key = f'severity_{i}_accuracy'
            severity_accuracies.append(
                self.evaluation_results.get(accuracy_key, 0)
            )
        
        bars = axes[1, 0].bar(severity_names, severity_accuracies, color=colors, alpha=0.7)
        axes[1, 0].set_title('Accuracy by Severity')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, severity_accuracies):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 5: Performance Metrics
        metrics = ['Accuracy', 'Severe Response Rate', 'False Alarm Rate']
        values = [
            self.evaluation_results.get('accuracy', 0),
            self.evaluation_results.get('severe_crash_response_rate', 0),
            1 - self.evaluation_results.get('false_alarm_rate', 0)  # Inverted for better visualization
        ]
        
        bars = axes[1, 1].bar(metrics, values, color=['blue', 'green', 'purple'], alpha=0.7)
        axes[1, 1].set_title('Key Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 6: Action Effectiveness (Mean Reward per Action)
        action_rewards = []
        for i in range(5):
            reward_key = f'action_{i}_mean_reward'
            action_rewards.append(
                self.evaluation_results.get(reward_key, 0)
            )
        
        bars = axes[1, 2].bar(range(5), action_rewards, color='lightcoral', alpha=0.7)
        axes[1, 2].set_title('Action Effectiveness')
        axes[1, 2].set_xlabel('Action')
        axes[1, 2].set_ylabel('Mean Reward')
        axes[1, 2].set_xticks(range(5))
        axes[1, 2].set_xticklabels([f'A{i}' for i in range(5)])
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, action_rewards):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to: {save_path}")
        
        plt.show()
    
    def generate_policy_heatmap(self, env_params: Dict = None, save_path: str = None):
        """
        Generate a heatmap showing policy behavior across different scenarios.
        
        Args:
            env_params: Parameters for environment creation
            save_path: Path to save the heatmap
        """
        if self.model is None:
            raise ValueError("No model loaded for heatmap generation")
        
        # Create test environment
        env_params = env_params or {'dataset_size': 1000, 'test_mode': False}
        test_env = CrashGuardEnv(**env_params)
        
        # Generate scenarios across different conditions
        scenarios = []
        actions_taken = []
        
        data_gen = CrashDataGenerator()
        
        # Create systematic scenarios
        speed_levels = [0.2, 0.5, 0.8]  # Low, medium, high speed
        gforce_levels = [0.1, 0.4, 0.8]  # Low, medium, high G-force
        drowsiness_levels = [0, 1]  # Alert, drowsy
        
        for speed in speed_levels:
            for gforce in gforce_levels:
                for drowsy in drowsiness_levels:
                    # Create scenario
                    scenario = {
                        'vehicle_speed': speed,
                        'impact_g_force': gforce,
                        'seatbelt_usage': 1,
                        'driver_drowsiness': drowsy,
                        'occupancy_count': 0.5,
                        'visibility_score': 0.7,
                        'road_type': 0.5,
                        'time_of_day': 0,
                        'crash_location_type': 0.5,
                        'distance_to_hospital': 0.3
                    }
                    
                    # Convert to state vector
                    state = np.array([
                        scenario['vehicle_speed'],
                        scenario['impact_g_force'],
                        scenario['seatbelt_usage'],
                        scenario['driver_drowsiness'],
                        scenario['occupancy_count'],
                        scenario['visibility_score'],
                        scenario['road_type'],
                        scenario['time_of_day'],
                        scenario['crash_location_type'],
                        scenario['distance_to_hospital']
                    ], dtype=np.float32)
                    
                    # Predict action
                    action, _ = self.model.predict(state, deterministic=True)
                    
                    scenarios.append((speed, gforce, drowsy))
                    actions_taken.append(action)
        
        # Create heatmap data
        heatmap_data = np.zeros((len(speed_levels) * len(gforce_levels), len(drowsiness_levels)))
        
        idx = 0
        for i, speed in enumerate(speed_levels):
            for j, gforce in enumerate(gforce_levels):
                for k, drowsy in enumerate(drowsiness_levels):
                    heatmap_data[i * len(gforce_levels) + j, k] = actions_taken[idx]
                    idx += 1
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        # Create labels
        scenario_labels = []
        for speed in speed_levels:
            for gforce in gforce_levels:
                scenario_labels.append(f'S:{speed:.1f}, G:{gforce:.1f}')
        
        drowsy_labels = ['Alert', 'Drowsy']
        
        sns.heatmap(heatmap_data, 
                   xticklabels=drowsy_labels,
                   yticklabels=scenario_labels,
                   annot=True, 
                   fmt='.0f',
                   cmap='viridis',
                   cbar_kws={'label': 'Action'})
        
        plt.title('Policy Action Heatmap\n(Speed, G-force vs Driver State)')
        plt.xlabel('Driver Drowsiness')
        plt.ylabel('Scenario (Speed, G-force)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Policy heatmap saved to: {save_path}")
        
        plt.show()
    
    def save_evaluation_results(self, path: str = None) -> str:
        """
        Save evaluation results to file.
        
        Args:
            path: Path to save results
            
        Returns:
            Path where results were saved
        """
        if not self.evaluation_results:
            print("No evaluation results to save")
            return
        
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"evaluation_results_{timestamp}.npz"
        
        np.savez(path, **self.evaluation_results)
        print(f"Evaluation results saved to: {path}")
        
        return path