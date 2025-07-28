import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random


class CrashDataGenerator:
    """
    Generates logical dummy dataset for crash scenarios with realistic correlations
    between features to simulate real-world crash detection scenarios.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the crash data generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Define feature ranges and distributions
        self.feature_ranges = {
            'vehicle_speed': (0, 120),  # km/h
            'impact_g_force': (0, 20),  # G units
            'seatbelt_usage': (0, 1),   # binary
            'driver_drowsiness': (0, 1), # binary
            'occupancy_count': (1, 8),   # passengers
            'visibility_score': (0, 1), # normalized
            'road_type': (0, 3),        # highway=0, rural=1, urban=2, residential=3
            'time_of_day': (0, 1),      # day=0, night=1
            'crash_location_type': (0, 3), # intersection=0, slope=1, curve=2, straight=3
            'distance_to_hospital': (0, 50) # km
        }
        
        # Define severity levels
        self.severity_levels = {
            'minor': 0,    # Low severity
            'moderate': 1, # Medium severity
            'severe': 2    # High severity
        }
    
    def generate_crash_scenario(self, severity: str = None) -> Dict:
        """
        Generate a single crash scenario with realistic feature correlations.
        
        Args:
            severity: Crash severity level ('minor', 'moderate', 'severe'), 
                     if None, randomly selected
                     
        Returns:
            Dictionary containing crash features and true severity
        """
        if severity is None:
            severity = random.choice(list(self.severity_levels.keys()))
        
        scenario = {}
        
        if severity == 'severe':
            # High severity crashes: high speed, high G-force, poor conditions
            scenario['vehicle_speed'] = np.random.normal(80, 15)
            scenario['impact_g_force'] = np.random.normal(12, 3)
            scenario['seatbelt_usage'] = np.random.choice([0, 1], p=[0.3, 0.7])
            scenario['driver_drowsiness'] = np.random.choice([0, 1], p=[0.4, 0.6])
            scenario['occupancy_count'] = np.random.randint(2, 6)
            scenario['visibility_score'] = np.random.beta(2, 5)  # Lower visibility
            scenario['road_type'] = np.random.choice([0, 1], p=[0.6, 0.4])  # Highway/rural
            scenario['time_of_day'] = np.random.choice([0, 1], p=[0.3, 0.7])  # Often at night
            scenario['crash_location_type'] = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
            scenario['distance_to_hospital'] = np.random.exponential(15)
            
        elif severity == 'moderate':
            # Moderate severity: medium speed, medium G-force
            scenario['vehicle_speed'] = np.random.normal(50, 12)
            scenario['impact_g_force'] = np.random.normal(6, 2)
            scenario['seatbelt_usage'] = np.random.choice([0, 1], p=[0.2, 0.8])
            scenario['driver_drowsiness'] = np.random.choice([0, 1], p=[0.7, 0.3])
            scenario['occupancy_count'] = np.random.randint(1, 4)
            scenario['visibility_score'] = np.random.beta(3, 3)  # Medium visibility
            scenario['road_type'] = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
            scenario['time_of_day'] = np.random.choice([0, 1], p=[0.6, 0.4])
            scenario['crash_location_type'] = np.random.choice([0, 2, 3], p=[0.3, 0.4, 0.3])
            scenario['distance_to_hospital'] = np.random.exponential(8)
            
        else:  # minor
            # Low severity: low speed, low G-force, good conditions
            scenario['vehicle_speed'] = np.random.normal(25, 8)
            scenario['impact_g_force'] = np.random.normal(2, 1)
            scenario['seatbelt_usage'] = np.random.choice([0, 1], p=[0.1, 0.9])
            scenario['driver_drowsiness'] = np.random.choice([0, 1], p=[0.9, 0.1])
            scenario['occupancy_count'] = np.random.randint(1, 3)
            scenario['visibility_score'] = np.random.beta(5, 2)  # Better visibility
            scenario['road_type'] = np.random.choice([2, 3], p=[0.6, 0.4])  # Urban/residential
            scenario['time_of_day'] = np.random.choice([0, 1], p=[0.8, 0.2])  # Mostly day
            scenario['crash_location_type'] = np.random.choice([2, 3], p=[0.3, 0.7])
            scenario['distance_to_hospital'] = np.random.exponential(5)
        
        # Clip values to valid ranges
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in scenario:
                scenario[feature] = np.clip(scenario[feature], min_val, max_val)
        
        # Add true severity label
        scenario['true_severity'] = self.severity_levels[severity]
        scenario['severity_label'] = severity
        
        return scenario
    
    def generate_dataset(self, n_samples: int = 10000, 
                        severity_distribution: Dict[str, float] = None) -> pd.DataFrame:
        """
        Generate a complete dataset of crash scenarios.
        
        Args:
            n_samples: Number of crash scenarios to generate
            severity_distribution: Distribution of severity levels 
                                  (default: balanced distribution)
                                  
        Returns:
            DataFrame containing generated crash scenarios
        """
        if severity_distribution is None:
            severity_distribution = {'minor': 0.5, 'moderate': 0.3, 'severe': 0.2}
        
        # Validate distribution
        assert abs(sum(severity_distribution.values()) - 1.0) < 1e-6, \
            "Severity distribution must sum to 1.0"
        
        scenarios = []
        
        for _ in range(n_samples):
            # Sample severity based on distribution
            severity = np.random.choice(
                list(severity_distribution.keys()),
                p=list(severity_distribution.values())
            )
            
            scenario = self.generate_crash_scenario(severity)
            scenarios.append(scenario)
        
        df = pd.DataFrame(scenarios)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features to [0, 1] range for RL environment.
        
        Args:
            df: DataFrame with crash scenarios
            
        Returns:
            DataFrame with normalized features
        """
        normalized_df = df.copy()
        
        # Normalize continuous features
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in normalized_df.columns and feature not in [
                'seatbelt_usage', 'driver_drowsiness', 'time_of_day'
            ]:  # Skip binary features
                normalized_df[feature] = (
                    (normalized_df[feature] - min_val) / (max_val - min_val)
                ).clip(0, 1)
        
        return normalized_df
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all features for documentation.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            'vehicle_speed': 'Vehicle speed at time of crash (0-120 km/h, normalized)',
            'impact_g_force': 'G-force from vibration sensor (0-20 G, normalized)',
            'seatbelt_usage': 'Seatbelt usage status (0=not used, 1=used)',
            'driver_drowsiness': 'Driver drowsiness status (0=alert, 1=drowsy)',
            'occupancy_count': 'Number of occupants (1-8, normalized)',
            'visibility_score': 'Weather visibility score (0=poor, 1=excellent)',
            'road_type': 'Road type (0=highway, 1=rural, 2=urban, 3=residential, normalized)',
            'time_of_day': 'Time of day (0=day, 1=night)',
            'crash_location_type': 'Location type (0=intersection, 1=slope, 2=curve, 3=straight, normalized)',
            'distance_to_hospital': 'Distance to nearest hospital (0-50 km, normalized)'
        }
    
    def get_action_descriptions(self) -> Dict[int, str]:
        """
        Get descriptions of all possible actions.
        
        Returns:
            Dictionary mapping action indices to descriptions
        """
        return {
            0: 'Send Immediate High-Priority Alert',
            1: 'Wait for Additional Confirmation', 
            2: 'Trigger Local Safety Mechanism (buzzer, hazard lights)',
            3: 'Broadcast Alert to Nearby Vehicles',
            4: 'Send Low-Priority Notification to Cloud'
        }