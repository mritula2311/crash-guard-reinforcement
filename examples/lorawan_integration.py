"""
Integration example for LoRaWAN/IoT systems with CrashGuard.

This example demonstrates how to integrate the trained CrashGuard model
with LoRaWAN-based IoT systems for real-time crash detection and response.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crash_guard import CrashGuardEvaluator


class LoRaWANInterface:
    """
    Simulated LoRaWAN interface for IoT communication.
    In a real implementation, this would interface with actual LoRaWAN modules.
    """
    
    def __init__(self, device_id: str = "CrashGuard_001"):
        self.device_id = device_id
        self.is_connected = False
        self.message_log = []
    
    def connect(self) -> bool:
        """Simulate connection to LoRaWAN network."""
        print(f"Connecting {self.device_id} to LoRaWAN network...")
        time.sleep(1)  # Simulate connection delay
        self.is_connected = True
        print("✓ Connected to LoRaWAN network")
        return True
    
    def send_message(self, message: Dict, priority: str = "normal") -> bool:
        """
        Send message via LoRaWAN.
        
        Args:
            message: Message payload
            priority: Message priority ("low", "normal", "high", "emergency")
            
        Returns:
            Success status
        """
        if not self.is_connected:
            print("❌ Error: Not connected to LoRaWAN network")
            return False
        
        # Add metadata
        lorawan_message = {
            "device_id": self.device_id,
            "timestamp": datetime.now().isoformat(),
            "priority": priority,
            "payload": message
        }
        
        # Simulate transmission delay based on priority
        delay_map = {"emergency": 0.1, "high": 0.3, "normal": 0.5, "low": 1.0}
        time.sleep(delay_map.get(priority, 0.5))
        
        # Log message
        self.message_log.append(lorawan_message)
        
        print(f"📡 LoRaWAN Message Sent ({priority} priority):")
        print(f"   {json.dumps(message, indent=2)}")
        
        return True
    
    def disconnect(self):
        """Disconnect from LoRaWAN network."""
        self.is_connected = False
        print("Disconnected from LoRaWAN network")


class CrashGuardIoTSystem:
    """
    Complete IoT-based crash detection system with RL-powered decision making.
    """
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = 'DQN',
                 device_id: str = "CrashGuard_001"):
        """
        Initialize the IoT crash detection system.
        
        Args:
            model_path: Path to trained RL model
            model_type: Type of RL model ('DQN' or 'PPO')
            device_id: Unique device identifier
        """
        self.device_id = device_id
        self.model_type = model_type
        
        # Initialize RL model
        print("Initializing CrashGuard AI model...")
        self.evaluator = CrashGuardEvaluator(model_path, model_type)
        
        # Initialize LoRaWAN interface
        self.lorawan = LoRaWANInterface(device_id)
        
        # System state
        self.system_active = False
        self.last_sensor_reading = None
        
        # Emergency contacts and systems
        self.emergency_contacts = {
            "hospital": "emergency@localhospital.com",
            "police": "dispatch@localpolice.com", 
            "family": "family@contact.com"
        }
        
        self.nearby_vehicles = []  # List of nearby vehicle IDs
        
    def start_system(self) -> bool:
        """Start the crash detection system."""
        print("🚀 Starting CrashGuard IoT System...")
        
        # Connect to LoRaWAN
        if not self.lorawan.connect():
            print("❌ Failed to connect to LoRaWAN network")
            return False
        
        self.system_active = True
        print("✅ CrashGuard system is now active and monitoring")
        
        # Send system startup notification
        startup_message = {
            "event_type": "system_startup",
            "device_id": self.device_id,
            "status": "active",
            "ai_model": self.model_type
        }
        self.lorawan.send_message(startup_message, "normal")
        
        return True
    
    def stop_system(self):
        """Stop the crash detection system."""
        print("🛑 Stopping CrashGuard system...")
        self.system_active = False
        self.lorawan.disconnect()
        print("✅ CrashGuard system stopped")
    
    def process_sensor_data(self, sensor_data: Dict) -> Optional[int]:
        """
        Process incoming sensor data and make crash response decision.
        
        Args:
            sensor_data: Dictionary containing sensor readings
            
        Returns:
            Action taken (0-4) or None if no action needed
        """
        if not self.system_active:
            print("⚠️ System not active")
            return None
        
        # Store sensor reading
        self.last_sensor_reading = sensor_data
        
        # Convert sensor data to normalized state vector
        state_vector = self._normalize_sensor_data(sensor_data)
        
        # Get AI decision
        action, confidence = self.evaluator.model.predict(state_vector, deterministic=True)
        
        print(f"🧠 AI Decision: Action {action} (confidence: {confidence})")
        
        # Execute action
        self._execute_action(action, sensor_data, confidence)
        
        return action
    
    def _normalize_sensor_data(self, sensor_data: Dict) -> np.ndarray:
        """
        Convert raw sensor data to normalized state vector for RL model.
        
        Args:
            sensor_data: Raw sensor readings
            
        Returns:
            Normalized state vector (10 features)
        """
        # Extract and normalize features
        # Note: In a real system, these would come from actual sensors
        
        # Vehicle speed (km/h) -> normalized to [0, 1]
        vehicle_speed = min(sensor_data.get('speed_kmh', 0) / 120.0, 1.0)
        
        # Impact G-force -> normalized to [0, 1] 
        impact_g_force = min(sensor_data.get('g_force', 0) / 20.0, 1.0)
        
        # Seatbelt usage (binary)
        seatbelt_usage = float(sensor_data.get('seatbelt_detected', False))
        
        # Driver drowsiness (binary)
        driver_drowsiness = float(sensor_data.get('driver_drowsy', False))
        
        # Occupancy count -> normalized
        occupancy_count = min(sensor_data.get('occupants', 1) / 8.0, 1.0)
        
        # Visibility score (from weather/camera)
        visibility_score = sensor_data.get('visibility', 1.0)
        
        # Road type (encoded and normalized)
        road_type_map = {'highway': 0, 'rural': 1, 'urban': 2, 'residential': 3}
        road_type = road_type_map.get(sensor_data.get('road_type', 'urban'), 2) / 3.0
        
        # Time of day (day=0, night=1)
        time_of_day = float(sensor_data.get('is_night', False))
        
        # Crash location type
        location_type_map = {'intersection': 0, 'slope': 1, 'curve': 2, 'straight': 3}
        crash_location_type = location_type_map.get(
            sensor_data.get('location_type', 'straight'), 3
        ) / 3.0
        
        # Distance to hospital (normalized)
        distance_to_hospital = min(sensor_data.get('hospital_distance_km', 10) / 50.0, 1.0)
        
        state_vector = np.array([
            vehicle_speed,
            impact_g_force,
            seatbelt_usage,
            driver_drowsiness,
            occupancy_count,
            visibility_score,
            road_type,
            time_of_day,
            crash_location_type,
            distance_to_hospital
        ], dtype=np.float32)
        
        return state_vector
    
    def _execute_action(self, action: int, sensor_data: Dict, confidence: float):
        """
        Execute the action determined by the RL model.
        
        Args:
            action: Action to execute (0-4)
            sensor_data: Original sensor data
            confidence: Model confidence
        """
        action_map = {
            0: self._send_high_priority_alert,
            1: self._wait_for_confirmation,
            2: self._trigger_local_safety,
            3: self._broadcast_to_vehicles,
            4: self._send_low_priority_notification
        }
        
        action_func = action_map.get(action)
        if action_func:
            action_func(sensor_data, confidence)
        else:
            print(f"❌ Unknown action: {action}")
    
    def _send_high_priority_alert(self, sensor_data: Dict, confidence: float):
        """Send immediate high-priority emergency alert."""
        print("🚨 EMERGENCY: Sending high-priority alert!")
        
        alert_message = {
            "event_type": "crash_detected",
            "severity": "severe",
            "location": {
                "latitude": sensor_data.get('latitude', 0.0),
                "longitude": sensor_data.get('longitude', 0.0),
                "address": sensor_data.get('address', 'Unknown')
            },
            "vehicle_info": {
                "speed": sensor_data.get('speed_kmh', 0),
                "occupants": sensor_data.get('occupants', 1),
                "g_force": sensor_data.get('g_force', 0)
            },
            "ai_confidence": float(confidence),
            "emergency_contacts": self.emergency_contacts,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send via LoRaWAN with emergency priority
        self.lorawan.send_message(alert_message, "emergency")
        
        # Also trigger local safety mechanisms
        self._trigger_local_safety(sensor_data, confidence, suppress_lorawan=True)
    
    def _wait_for_confirmation(self, sensor_data: Dict, confidence: float):
        """Wait for additional sensor confirmation before acting."""
        print("⏳ Waiting for additional sensor confirmation...")
        
        confirmation_message = {
            "event_type": "incident_detected",
            "status": "monitoring",
            "sensor_data": sensor_data,
            "ai_confidence": float(confidence),
            "action": "awaiting_confirmation"
        }
        
        self.lorawan.send_message(confirmation_message, "normal")
    
    def _trigger_local_safety(self, sensor_data: Dict, confidence: float, suppress_lorawan: bool = False):
        """Trigger local safety mechanisms (buzzer, hazard lights, etc.)."""
        print("🔊 Activating local safety mechanisms...")
        print("   - Emergency buzzer activated")
        print("   - Hazard lights activated") 
        print("   - Doors unlocked")
        print("   - Emergency beacon activated")
        
        if not suppress_lorawan:
            safety_message = {
                "event_type": "local_safety_activated",
                "mechanisms": ["buzzer", "hazard_lights", "door_unlock", "beacon"],
                "sensor_data": sensor_data,
                "ai_confidence": float(confidence)
            }
            
            self.lorawan.send_message(safety_message, "high")
    
    def _broadcast_to_vehicles(self, sensor_data: Dict, confidence: float):
        """Broadcast warning to nearby vehicles."""
        print("📢 Broadcasting warning to nearby vehicles...")
        
        broadcast_message = {
            "event_type": "traffic_warning",
            "warning_type": "accident_ahead", 
            "location": {
                "latitude": sensor_data.get('latitude', 0.0),
                "longitude": sensor_data.get('longitude', 0.0)
            },
            "details": {
                "estimated_severity": "moderate_to_severe",
                "recommended_action": "reduce_speed_change_lane"
            },
            "ai_confidence": float(confidence),
            "range_meters": 1000
        }
        
        self.lorawan.send_message(broadcast_message, "high")
        
        # Also send to cloud for traffic management
        self._send_low_priority_notification(sensor_data, confidence, suppress_lorawan=True)
    
    def _send_low_priority_notification(self, sensor_data: Dict, confidence: float, suppress_lorawan: bool = False):
        """Send low-priority notification to cloud systems."""
        print("☁️ Sending notification to cloud systems...")
        
        if not suppress_lorawan:
            cloud_message = {
                "event_type": "incident_report",
                "severity": "minor",
                "sensor_data": sensor_data,
                "ai_confidence": float(confidence),
                "follow_up_required": False
            }
            
            self.lorawan.send_message(cloud_message, "low")


def simulate_crash_scenarios():
    """Simulate various crash scenarios for demonstration."""
    
    # Note: In a real deployment, you would load an actual trained model
    print("🚨 CRASHGUARD LORAWAN INTEGRATION DEMO")
    print("="*60)
    print("Note: This demo uses simulated components.")
    print("In a real deployment, replace with actual:")
    print("- Trained RL model")
    print("- LoRaWAN hardware interface")  
    print("- Real sensor data")
    print("="*60)
    
    # For demo purposes, create a mock system
    # In real usage: system = CrashGuardIoTSystem("path/to/trained/model.zip", "DQN")
    
    # Simulate system without actual model for demo
    mock_system = type('MockSystem', (), {})()
    mock_system.device_id = "CrashGuard_Demo_001"
    mock_system.lorawan = LoRaWANInterface("CrashGuard_Demo_001")
    mock_system.system_active = False
    
    # Start system
    print("\n1. Starting IoT System...")
    mock_system.lorawan.connect()
    mock_system.system_active = True
    
    # Simulate different crash scenarios
    crash_scenarios = [
        {
            "name": "Minor Fender Bender",
            "sensor_data": {
                "speed_kmh": 15,
                "g_force": 1.5,
                "seatbelt_detected": True,
                "driver_drowsy": False,
                "occupants": 1,
                "visibility": 0.9,
                "road_type": "urban",
                "is_night": False,
                "location_type": "intersection",
                "hospital_distance_km": 3.2,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "address": "Broadway & 42nd St, NYC"
            },
            "expected_action": "Local Safety Mechanism"
        },
        {
            "name": "Highway Collision",
            "sensor_data": {
                "speed_kmh": 95,
                "g_force": 15.8,
                "seatbelt_detected": True,
                "driver_drowsy": False,
                "occupants": 3,
                "visibility": 0.4,
                "road_type": "highway",
                "is_night": True,
                "location_type": "straight",
                "hospital_distance_km": 12.5,
                "latitude": 40.7589,
                "longitude": -73.9851,
                "address": "I-95 North, Mile Marker 45"
            },
            "expected_action": "High-Priority Alert"
        },
        {
            "name": "Rural Road Incident",
            "sensor_data": {
                "speed_kmh": 55,
                "g_force": 8.2,
                "seatbelt_detected": False,
                "driver_drowsy": True,
                "occupants": 2,
                "visibility": 0.6,
                "road_type": "rural",
                "is_night": True,
                "location_type": "curve",
                "hospital_distance_km": 25.0,
                "latitude": 41.8781,
                "longitude": -87.6298,
                "address": "County Road 45, Rural Area"
            },
            "expected_action": "Broadcast to Vehicles"
        }
    ]
    
    print(f"\n2. Simulating {len(crash_scenarios)} crash scenarios...")
    
    for i, scenario in enumerate(crash_scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        print(f"Sensor Data: {json.dumps(scenario['sensor_data'], indent=2)}")
        
        # Simulate action based on scenario characteristics
        sensor_data = scenario['sensor_data']
        
        if sensor_data['g_force'] > 10 or sensor_data['speed_kmh'] > 80:
            action = 0  # High-priority alert
            action_name = "High-Priority Alert"
        elif sensor_data['g_force'] > 5 or sensor_data['driver_drowsy']:
            action = 3  # Broadcast to vehicles  
            action_name = "Broadcast to Vehicles"
        else:
            action = 2  # Local safety mechanism
            action_name = "Local Safety Mechanism"
        
        print(f"🧠 AI Decision: {action_name}")
        
        # Simulate message sending
        if action == 0:
            message = {
                "event_type": "crash_detected",
                "severity": "severe",
                "location": {
                    "latitude": sensor_data['latitude'],
                    "longitude": sensor_data['longitude'],
                    "address": sensor_data['address']
                }
            }
            mock_system.lorawan.send_message(message, "emergency")
            
        elif action == 3:
            message = {
                "event_type": "traffic_warning",
                "warning_type": "accident_ahead",
                "location": {
                    "latitude": sensor_data['latitude'],
                    "longitude": sensor_data['longitude']
                }
            }
            mock_system.lorawan.send_message(message, "high")
            
        elif action == 2:
            message = {
                "event_type": "local_safety_activated",
                "mechanisms": ["buzzer", "hazard_lights"]
            }
            mock_system.lorawan.send_message(message, "high")
        
        time.sleep(2)  # Pause between scenarios
    
    print(f"\n3. Demo completed. Total messages sent: {len(mock_system.lorawan.message_log)}")
    
    # Show message log
    print("\n4. LoRaWAN Message Log:")
    for i, msg in enumerate(mock_system.lorawan.message_log, 1):
        print(f"   Message {i} ({msg['priority']} priority):")
        print(f"     Event: {msg['payload'].get('event_type', 'unknown')}")
        print(f"     Time: {msg['timestamp']}")
    
    mock_system.lorawan.disconnect()


if __name__ == "__main__":
    simulate_crash_scenarios()