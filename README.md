# CrashGuard: Intelligent IoT-based Vehicle Accident Detection System

🚗 **CrashGuard** is a reinforcement learning (RL) environment that models crash severity estimation and real-time response decision-making for intelligent IoT-based vehicle accident detection systems.

## 🎯 Overview

CrashGuard uses deep reinforcement learning to predict crash severity and select optimal response actions based on observed crash features including internal vehicle dynamics, human behavior, and external conditions. The system is designed for real-time deployment in smart mobility platforms, especially in remote areas with LoRaWAN-based communication where cellular networks are unavailable.

## 🧠 Key Features

- **Advanced RL Environment**: OpenAI Gym-compatible environment with realistic crash scenarios
- **Multi-modal State Space**: 10 normalized features including vehicle dynamics, human factors, and environmental conditions
- **Intelligent Action Space**: 5 discrete response actions from local safety mechanisms to emergency alerts
- **Realistic Reward Function**: Balances emergency response efficiency with false alarm minimization
- **LoRaWAN Integration**: Ready-to-deploy IoT system integration with example implementations
- **Comprehensive Evaluation**: Performance metrics, learning curves, and policy visualization tools

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/mritula2311/crash-guard.git
cd crash-guard

# Install dependencies
pip install -r requirements.txt

# Run basic demo
python examples/basic_usage.py
```

## 🚀 Quick Start

### 1. Train a Model
```bash
# Quick training with default parameters
python train.py --quick --model DQN --timesteps 50000

# Custom training
python train.py --model PPO --timesteps 100000 --dataset-size 10000 --learning-rate 0.001
```

### 2. Evaluate a Model
```bash
# Basic evaluation
python evaluate.py --model-path logs/training_dqn_20250728_062109/crash_guard_dqn_20250728_062111.zip --episodes 1000

# Comprehensive evaluation with heatmap
python evaluate.py --model-path logs/training_dqn_20250728_062109/crash_guard_dqn_20250728_062111.zip --episodes 1000 --generate-heatmap
```

### 3. Explore the Environment
```python
from crash_guard import CrashGuardEnv

# Create environment
env = CrashGuardEnv(dataset_size=1000, random_seed=42)

# Run episode
obs = env.reset()
action = env.action_space.sample()  # Random action
next_obs, reward, done, info = env.step(action)

# Render scenario
env.render()
```

## 📊 Environment Specifications

### State Space (10 features, normalized to [0,1])

| Feature | Description | Range |
|---------|-------------|-------|
| `vehicle_speed` | Vehicle speed at crash | 0-120 km/h |
| `impact_g_force` | G-force from vibration sensor | 0-20 G |
| `seatbelt_usage` | Seatbelt usage status | Binary (0/1) |
| `driver_drowsiness` | Driver drowsiness detection | Binary (0/1) |
| `occupancy_count` | Number of vehicle occupants | 1-8 people |
| `visibility_score` | Weather visibility score | 0-1 (poor to excellent) |
| `road_type` | Road type classification | Highway/Rural/Urban/Residential |
| `time_of_day` | Time of incident | Binary (Day/Night) |
| `crash_location_type` | Location geometry | Intersection/Slope/Curve/Straight |
| `distance_to_hospital` | Distance to nearest hospital | 0-50 km |

### Action Space (5 discrete actions)

| Action ID | Description | Use Case |
|-----------|-------------|----------|
| 0 | Send Immediate High-Priority Alert | Severe crashes requiring immediate response |
| 1 | Wait for Additional Confirmation | Uncertain scenarios needing more data |
| 2 | Trigger Local Safety Mechanism | Moderate crashes, activate hazard systems |
| 3 | Broadcast Alert to Nearby Vehicles | Traffic warnings for accident prevention |
| 4 | Send Low-Priority Notification to Cloud | Minor incidents for record keeping |

### Reward Function

- **+10**: Correct high-severity crash identification with timely alert
- **+5**: Correct low-severity classification with appropriate response
- **-10**: False high-priority alert (emergency resource waste)
- **-5**: Delayed response when drowsiness or high G-force detected
- **Additional penalties**: For ignoring severe crashes

## 📈 Performance Metrics

The system tracks comprehensive metrics including:

- **Accuracy**: Overall decision accuracy across all scenarios
- **False Alarm Rate**: Frequency of unnecessary emergency alerts
- **Severe Crash Response Rate**: Appropriate response to high-severity incidents
- **Action Distribution**: Usage patterns across different response types
- **Severity-Specific Performance**: Performance breakdown by crash severity

## 🌐 IoT Integration

### LoRaWAN Deployment Example

```python
from crash_guard import CrashGuardEvaluator
from examples.lorawan_integration import CrashGuardIoTSystem

# Initialize system with trained model
system = CrashGuardIoTSystem("path/to/trained_model.zip", "DQN")

# Start monitoring
system.start_system()

# Process sensor data
sensor_data = {
    "speed_kmh": 75,
    "g_force": 12.5,
    "seatbelt_detected": True,
    "driver_drowsy": False,
    "occupants": 2,
    "visibility": 0.6,
    "road_type": "highway",
    "is_night": True,
    "hospital_distance_km": 8.3
}

# AI makes decision and triggers appropriate response
action = system.process_sensor_data(sensor_data)
```

### Integration Guidelines

1. **Sensor Interface**: Connect accelerometers, cameras, and environmental sensors
2. **LoRaWAN Setup**: Configure LoRaWAN module for low-power, long-range communication
3. **Edge Computing**: Deploy trained model on edge device for real-time inference
4. **Cloud Backend**: Set up cloud services for data logging and system monitoring
5. **Emergency Integration**: Connect to local emergency services and traffic management

## 📁 Project Structure

```
crash-guard/
├── crash_guard/                 # Main package
│   ├── __init__.py              # Package initialization
│   ├── environment.py           # RL environment implementation
│   ├── data_generator.py        # Crash scenario data generation
│   ├── trainer.py               # Model training utilities
│   └── evaluator.py             # Model evaluation and visualization
├── examples/                    # Example implementations
│   ├── basic_usage.py           # Basic environment demonstration
│   └── lorawan_integration.py   # IoT system integration example
├── train.py                     # Main training script
├── evaluate.py                  # Main evaluation script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🧪 Examples and Demos

### Basic Environment Usage
```bash
python examples/basic_usage.py
```

### LoRaWAN Integration Demo
```bash
python examples/lorawan_integration.py
```

### Custom Training Pipeline
```python
from crash_guard import CrashGuardTrainer

# Initialize trainer
trainer = CrashGuardTrainer(
    env_params={'dataset_size': 5000, 'random_seed': 42},
    model_type='DQN',
    model_params={'learning_rate': 0.001}
)

# Train model
trainer.create_environment()
trainer.create_model()
model_path = trainer.train(total_timesteps=100000)

# Generate training plots
trainer.plot_training_progress('training_progress.png')
```

## 📊 Evaluation and Visualization

The system provides comprehensive evaluation tools:

- **Performance Metrics**: Accuracy, precision, recall by severity level
- **Learning Curves**: Training progress and convergence visualization
- **Policy Heatmaps**: Decision patterns across different scenarios
- **Action Distribution**: Analysis of response action usage
- **Reward Analysis**: Detailed reward breakdown by scenario type

## 🔧 Configuration Options

### Training Configuration
- **Model Types**: DQN, PPO (extensible to other algorithms)
- **Dataset Size**: Configurable number of training scenarios
- **Hyperparameters**: Learning rate, batch size, exploration parameters
- **Reward Tuning**: Adjustable reward weights for different outcomes

### Environment Configuration
- **Severity Distribution**: Customizable crash severity ratios
- **Feature Ranges**: Adjustable sensor value ranges
- **Scenario Complexity**: Variable complexity crash scenarios
- **Test Modes**: Deterministic evaluation modes

## 🚨 Safety Considerations

1. **Fail-Safe Design**: System defaults to alerting in uncertain situations
2. **Redundancy**: Multiple confirmation mechanisms for critical decisions
3. **Privacy**: Local processing minimizes sensitive data transmission
4. **Reliability**: Robust error handling and fallback mechanisms
5. **Compliance**: Designed for automotive safety standards compatibility

## 🤝 Contributing

We welcome contributions! Please consider:

1. **Bug Reports**: Submit detailed bug reports with reproduction steps
2. **Feature Requests**: Propose new features with use case descriptions
3. **Code Contributions**: Follow coding standards and include tests
4. **Documentation**: Help improve documentation and examples
5. **Integration Examples**: Share real-world deployment experiences

## 📜 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Gym for the RL environment framework
- Stable-Baselines3 for robust RL algorithm implementations
- LoRaWAN Alliance for IoT communication standards
- Automotive safety research community for domain expertise

## 📞 Support

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: [Create an issue](https://github.com/mritula2311/crash-guard/issues)
- **Documentation**: Check examples and inline documentation
- **Community**: Join discussions in GitHub Discussions

---

**CrashGuard** - Saving lives through intelligent crash detection and response 🚗🧠🚨