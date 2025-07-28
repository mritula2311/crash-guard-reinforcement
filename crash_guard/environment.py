import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .data_generator import CrashDataGenerator


class CrashGuardEnv(gym.Env):
    """
    OpenAI Gym environment for crash severity estimation and real-time response
    decision-making in intelligent IoT-based vehicle accident detection system.
    """
    
    def __init__(self, 
                 dataset_size: int = 10000,
                 test_mode: bool = False,
                 random_seed: int = 42):
        """
        Initialize the CrashGuard RL environment.
        
        Args:
            dataset_size: Size of the crash scenario dataset
            test_mode: If True, use deterministic scenarios for evaluation
            random_seed: Random seed for reproducibility
        """
        super(CrashGuardEnv, self).__init__()
        
        self.dataset_size = dataset_size
        self.test_mode = test_mode
        self.random_seed = random_seed
        
        # Initialize data generator
        self.data_generator = CrashDataGenerator(random_seed)
        
        # Generate dataset
        self.dataset = self._generate_dataset()
        self.current_scenario_idx = 0
        self.current_scenario = None
        
        # Define state space (10 normalized features)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(10,), 
            dtype=np.float32
        )
        
        # Define action space (5 discrete actions)
        self.action_space = spaces.Discrete(5)
        
        # Track episode statistics
        self.episode_count = 0
        self.total_reward = 0
        self.correct_predictions = 0
        self.false_alarms = 0
        self.missed_severe_crashes = 0
        
        # Feature names for reference
        self.feature_names = [
            'vehicle_speed', 'impact_g_force', 'seatbelt_usage', 'driver_drowsiness',
            'occupancy_count', 'visibility_score', 'road_type', 'time_of_day',
            'crash_location_type', 'distance_to_hospital'
        ]
    
    def _generate_dataset(self) -> pd.DataFrame:
        """Generate and normalize the crash scenario dataset."""
        # Generate raw dataset
        raw_dataset = self.data_generator.generate_dataset(
            n_samples=self.dataset_size,
            severity_distribution={'minor': 0.5, 'moderate': 0.3, 'severe': 0.2}
        )
        
        # Normalize features
        normalized_dataset = self.data_generator.normalize_features(raw_dataset)
        
        return normalized_dataset
    
    def _get_current_state(self) -> np.ndarray:
        """
        Get the current state representation.
        
        Returns:
            Normalized state vector (10 features)
        """
        if self.current_scenario is None:
            return np.zeros(10, dtype=np.float32)
        
        state = np.array([
            self.current_scenario['vehicle_speed'],
            self.current_scenario['impact_g_force'],
            self.current_scenario['seatbelt_usage'],
            self.current_scenario['driver_drowsiness'],
            self.current_scenario['occupancy_count'],
            self.current_scenario['visibility_score'],
            self.current_scenario['road_type'],
            self.current_scenario['time_of_day'],
            self.current_scenario['crash_location_type'],
            self.current_scenario['distance_to_hospital']
        ], dtype=np.float32)
        
        return state
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward based on action taken and true crash severity.
        
        Args:
            action: Action taken by the agent (0-4)
            
        Returns:
            Reward value
        """
        if self.current_scenario is None:
            return 0.0
        
        true_severity = self.current_scenario['true_severity']
        drowsiness = self.current_scenario['driver_drowsiness']
        g_force = self.current_scenario['impact_g_force']
        
        reward = 0.0
        
        # Action mappings:
        # 0: Send Immediate High-Priority Alert
        # 1: Wait for Additional Confirmation
        # 2: Trigger Local Safety Mechanism
        # 3: Broadcast Alert to Nearby Vehicles
        # 4: Send Low-Priority Notification to Cloud
        
        if action == 0:  # High-Priority Alert
            if true_severity == 2:  # Severe crash
                reward = 10.0  # Correct high-priority alert
                self.correct_predictions += 1
            elif true_severity == 1:  # Moderate crash
                reward = 2.0   # Acceptable but not optimal
            else:  # Minor crash
                reward = -10.0  # False alarm - wasted emergency resources
                self.false_alarms += 1
                
        elif action == 1:  # Wait for confirmation
            if true_severity == 2:  # Severe crash
                # Penalty for delay, especially with drowsiness or high G-force
                base_penalty = -5.0
                if drowsiness == 1:
                    base_penalty -= 2.0
                if g_force > 0.6:  # High G-force (normalized)
                    base_penalty -= 2.0
                reward = base_penalty
                self.missed_severe_crashes += 1
            elif true_severity == 1:  # Moderate crash
                reward = 1.0   # Reasonable to wait for moderate crashes
            else:  # Minor crash
                reward = 3.0   # Good decision to wait
                
        elif action == 2:  # Local Safety Mechanism
            if true_severity == 2:  # Severe crash
                reward = 3.0   # Helpful but not sufficient for severe crashes
            elif true_severity == 1:  # Moderate crash
                reward = 6.0   # Good response for moderate crashes
                self.correct_predictions += 1
            else:  # Minor crash
                reward = 4.0   # Appropriate response
                self.correct_predictions += 1
                
        elif action == 3:  # Broadcast to nearby vehicles
            if true_severity == 2:  # Severe crash
                reward = 7.0   # Good preventive action
                self.correct_predictions += 1
            elif true_severity == 1:  # Moderate crash
                reward = 5.0   # Helpful for moderate crashes
                self.correct_predictions += 1
            else:  # Minor crash
                reward = 2.0   # Less critical but still useful
                
        elif action == 4:  # Low-Priority Cloud Notification
            if true_severity == 2:  # Severe crash
                reward = -8.0  # Inadequate response for severe crashes
                self.missed_severe_crashes += 1
            elif true_severity == 1:  # Moderate crash
                reward = 1.0   # Minimal but acceptable
            else:  # Minor crash
                reward = 5.0   # Correct low-priority response
                self.correct_predictions += 1
        
        # Additional penalties for ignoring severe crashes
        if true_severity == 2 and action not in [0, 3]:  # Not high-priority or broadcast
            reward -= 3.0
        
        return reward
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (initial state observation, info dict)
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Select next scenario
        if self.test_mode:
            # In test mode, cycle through scenarios deterministically
            self.current_scenario_idx = self.episode_count % len(self.dataset)
        else:
            # In training mode, select random scenario
            self.current_scenario_idx = np.random.randint(0, len(self.dataset))
        
        self.current_scenario = self.dataset.iloc[self.current_scenario_idx].to_dict()
        
        # Reset episode statistics
        self.total_reward = 0
        
        info = {}
        
        return self._get_current_state(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-4)
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        # Calculate reward
        reward = self._calculate_reward(action)
        self.total_reward += reward
        
        # Episode is done after one step (single decision per crash scenario)
        terminated = True
        truncated = False  # Episode never truncated in this environment
        
        # Collect info
        info = {
            'true_severity': self.current_scenario['true_severity'],
            'severity_label': self.current_scenario['severity_label'],
            'action_taken': action,
            'reward': reward,
            'episode_reward': self.total_reward,
            'vehicle_speed': self.current_scenario['vehicle_speed'],
            'impact_g_force': self.current_scenario['impact_g_force'],
            'driver_drowsiness': self.current_scenario['driver_drowsiness']
        }
        
        # Update episode count
        self.episode_count += 1
        
        # Next state is the same (episode ends)
        next_state = self._get_current_state()
        
        return next_state, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            String representation of current state or None
        """
        if self.current_scenario is None:
            return "No active scenario"
        
        output = f"\n=== CrashGuard Scenario {self.episode_count} ===\n"
        output += f"True Severity: {self.current_scenario['severity_label']} "
        output += f"(Level {self.current_scenario['true_severity']})\n"
        
        output += "\nCrash Context:\n"
        for i, feature in enumerate(self.feature_names):
            value = self.current_scenario[feature]
            if feature in ['seatbelt_usage', 'driver_drowsiness', 'time_of_day']:
                # Binary features
                labels = {
                    'seatbelt_usage': ['Not Used', 'Used'],
                    'driver_drowsiness': ['Alert', 'Drowsy'], 
                    'time_of_day': ['Day', 'Night']
                }
                output += f"  {feature}: {labels[feature][int(value)]}\n"
            elif feature == 'road_type':
                road_types = ['Highway', 'Rural', 'Urban', 'Residential']
                idx = min(int(value * 4), 3)
                output += f"  {feature}: {road_types[idx]}\n"
            elif feature == 'crash_location_type':
                location_types = ['Intersection', 'Slope', 'Curve', 'Straight']
                idx = min(int(value * 4), 3)
                output += f"  {feature}: {location_types[idx]}\n"
            else:
                output += f"  {feature}: {value:.3f}\n"
        
        output += f"\nEpisode Statistics:\n"
        output += f"  Total Episodes: {self.episode_count}\n"
        output += f"  Correct Predictions: {self.correct_predictions}\n"
        output += f"  False Alarms: {self.false_alarms}\n"
        output += f"  Missed Severe Crashes: {self.missed_severe_crashes}\n"
        
        if mode == 'human':
            print(output)
        
        return output
    
    def get_statistics(self) -> Dict:
        """
        Get environment statistics for evaluation.
        
        Returns:
            Dictionary containing performance statistics
        """
        total_episodes = max(self.episode_count, 1)
        
        return {
            'total_episodes': self.episode_count,
            'correct_predictions': self.correct_predictions,
            'false_alarms': self.false_alarms,
            'missed_severe_crashes': self.missed_severe_crashes,
            'accuracy': self.correct_predictions / total_episodes,
            'false_alarm_rate': self.false_alarms / total_episodes,
            'missed_severe_rate': self.missed_severe_crashes / total_episodes
        }
    
    def close(self):
        """Clean up environment resources."""
        pass