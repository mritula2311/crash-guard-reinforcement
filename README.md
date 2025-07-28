# CrashGuard: Intelligent IoT-based Vehicle Accident Detection System

🚗 **CrashGuard** is a reinforcement learning (RL) environment that models crash severity estimation and real-time response decision-making for intelligent IoT-based vehicle accident detection systems.

## 🎯 Overview

CrashGuard uses deep reinforcement learning to predict crash severity and select optimal response actions based on observed crash features including internal vehicle dynamics, human behavior, and external conditions. The system is designed for real-time deployment in smart mobility platforms, especially in remote areas with LoRaWAN-based communication where cellular networks are unavailable.

## 🧠 Key Features

- **Advanced RL Environment**: Gymnasium-compatible environment with realistic crash scenarios
- **Multi-modal State Space**: 10 normalized features including vehicle dynamics, human factors, and environmental conditions
- **Intelligent Action Space**: 5 discrete response actions from local safety mechanisms to emergency alerts
- **Enhanced Reward Shaping**: Contextual reward function with accuracy bonuses for improved learning
- **Balanced Training**: Equal crash severity distribution to prevent model bias
- **Curriculum Learning**: Progressive difficulty increase for better convergence
- **Hyperparameter Optimization**: Automated tuning for maximum accuracy
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

**Note**: CrashGuard now uses Gymnasium (the maintained successor to OpenAI Gym) for better compatibility and performance.

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

### 🎯 For Immediate Accuracy Improvement
```bash
# If you have a model with poor accuracy (like 60%), run this first:
python improve_accuracy.py

# This addresses common issues:
# - Models using only one action type
# - Poor minor crash detection
# - Low overall accuracy
```

### 1. Train a Model
```bash
# Enhanced training with improved accuracy (Recommended)
python train.py --enhanced --model DQN --timesteps 100000 --dataset-size 15000

# Advanced enhanced training with all features
python train_enhanced.py --model DQN --timesteps 100000 --balanced-training --enhanced-rewards

# Quick training with default parameters
python train.py --quick --model DQN --timesteps 50000

# Custom training with specific parameters
python train.py --model PPO --timesteps 100000 --dataset-size 10000 --learning-rate 0.001
```

### 2. Improve Model Accuracy
```bash
# Automatic hyperparameter optimization for best results
python optimize_hyperparameters.py --model DQN --n-trials 50

# Quick accuracy improvement (addresses common issues)
python improve_accuracy.py

# Train with curriculum learning for progressive difficulty
python train_enhanced.py --model DQN --curriculum-learning --balanced-training
```

### 2. Evaluate a Model
```bash
# Basic evaluation with available enhanced models
python evaluate.py --model-path logs/enhanced_training_*/best_model.zip --episodes 1000

# Comprehensive evaluation with heatmap and visualizations
python evaluate.py --model-path logs/enhanced_training_*/best_model.zip --episodes 1000 --generate-heatmap

# Compare different models
python evaluate.py --model-path logs/training_dqn_*/crash_guard_dqn_*.zip --episodes 500
```

### 3. Explore the Environment
```python
from crash_guard import CrashGuardEnv

# Create enhanced environment with balanced training
env = CrashGuardEnv(
    dataset_size=1000, 
    random_seed=42,
    balanced_severity=True,      # Equal crash type distribution
    enhanced_rewards=True        # Better reward shaping
)

# Run episode
obs, info = env.reset()
action = env.action_space.sample()  # Random action
next_obs, reward, done, truncated, info = env.step(action)

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

#### Enhanced Reward System (New!)
- **+15**: Correct severe crash identification with immediate response
- **+8**: Appropriate moderate crash handling
- **+6**: Correct minor crash classification
- **-12**: False high-priority alert (emergency resource waste)
- **-8**: Delayed response to severe crashes
- **Contextual Bonuses**: Additional rewards based on G-force, speed, drowsiness detection
- **Accuracy Bonuses**: Extra rewards for maintaining high overall accuracy
- **Action Diversity**: Encourages learning multiple appropriate responses

#### Standard Reward System
- **+10**: Correct high-severity crash identification with timely alert
- **+5**: Correct low-severity classification with appropriate response
- **-10**: False high-priority alert (emergency resource waste)
- **-5**: Delayed response when drowsiness or high G-force detected
- **Additional penalties**: For ignoring severe crashes

## 📈 Performance Metrics

The system tracks comprehensive metrics including:

- **Overall Accuracy**: Decision accuracy across all scenarios (Target: 85%+)
- **Severe Crash Response Rate**: Appropriate response to high-severity incidents (Target: 95%+)
- **False Alarm Rate**: Frequency of unnecessary emergency alerts (Target: <8%)
- **Action Distribution**: Usage patterns across different response types
- **Severity-Specific Performance**: Performance breakdown by crash severity
- **Training Convergence**: Learning curves and stability metrics
- **Action Diversity**: Measures healthy exploration vs exploitation balance

### Expected Performance with Enhanced Training

| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| **Overall Accuracy** | ~60-75% | **85-90%** | **+10-15%** |
| **Severe Crash Detection** | ~80% | **92-95%** | **+12-15%** |
| **Minor Crash Accuracy** | ~40% | **70-80%** | **+30-40%** |
| **False Alarm Rate** | ~15% | **5-8%** | **-7-10%** |
| **Action Diversity** | 1-2 actions | **3-4 actions** | **Much better** |

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
│   ├── environment.py           # Enhanced RL environment implementation
│   ├── data_generator.py        # Crash scenario data generation
│   ├── trainer.py               # Model training utilities
│   └── evaluator.py             # Model evaluation and visualization
├── examples/                    # Example implementations
│   ├── basic_usage.py           # Basic environment demonstration
│   └── lorawan_integration.py   # IoT system integration example
├── train.py                     # Main training script (enhanced)
├── train_enhanced.py            # Advanced training with all improvements
├── improve_accuracy.py          # Quick accuracy improvement script
├── optimize_hyperparameters.py  # Automated hyperparameter tuning
├── evaluate.py                  # Main evaluation script
├── requirements.txt             # Python dependencies (updated)
├── ACCURACY_IMPROVEMENT_GUIDE.md # Detailed improvement guide
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

### Enhanced Training Pipeline
```python
from crash_guard import CrashGuardTrainer

# Initialize enhanced trainer
trainer = CrashGuardTrainer(
    env_params={
        'dataset_size': 15000, 
        'random_seed': 42,
        'balanced_severity': True,    # Better training balance
        'enhanced_rewards': True      # Improved reward shaping
    },
    model_type='DQN',
    model_params={
        'learning_rate': 5e-4,
        'batch_size': 128,
        'exploration_fraction': 0.4   # Enhanced exploration
    }
)

# Train enhanced model
trainer.create_environment()
trainer.create_model()
model_path = trainer.train(total_timesteps=100000)

# Generate training plots
trainer.plot_training_progress('training_progress.png')
```

### Accuracy Improvement Workflow
```bash
# Step 1: Quick accuracy improvement
python improve_accuracy.py

# Step 2: Hyperparameter optimization (optional, for best results)
python optimize_hyperparameters.py --model DQN --n-trials 30

# Step 3: Enhanced training with optimized parameters
python train_enhanced.py --model DQN --balanced-training --curriculum-learning

# Step 4: Comprehensive evaluation
python evaluate.py --model-path logs/enhanced_*/best_model.zip --episodes 1000 --generate-heatmap
```

## 📊 Evaluation and Visualization

The system provides comprehensive evaluation tools:

- **Performance Metrics**: Accuracy, precision, recall by severity level
- **Learning Curves**: Training progress and convergence visualization
- **Policy Heatmaps**: Decision patterns across different scenarios
- **Action Distribution Analysis**: Response action usage patterns
- **Reward Analysis**: Detailed reward breakdown by scenario type
- **Comparative Analysis**: Compare multiple model performances
- **Real-time Training Monitoring**: Evaluation callbacks during training

### Enhanced Evaluation Features
- **Severity-Specific Analysis**: Detailed breakdown by crash type
- **Action Diversity Metrics**: Measure of decision-making breadth
- **False Alarm Analysis**: Specific false positive pattern detection
- **Convergence Diagnostics**: Training stability and learning progress
- **Hyperparameter Impact**: Visualization of parameter effects on performance

## 🔧 Configuration Options

### Training Configuration
- **Model Types**: DQN, PPO (with enhanced architectures)
- **Dataset Size**: Configurable number of training scenarios (recommended: 15,000+)
- **Hyperparameters**: Learning rate, batch size, exploration parameters
- **Reward Tuning**: Standard vs Enhanced reward shaping
- **Training Features**: Balanced training, curriculum learning, hyperparameter optimization

### Environment Configuration
- **Severity Distribution**: Customizable crash severity ratios or balanced mode
- **Feature Ranges**: Adjustable sensor value ranges
- **Scenario Complexity**: Variable complexity crash scenarios
- **Test Modes**: Deterministic evaluation modes
- **Enhanced Features**: Enhanced rewards, curriculum learning, balanced datasets

### Advanced Options
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Multi-Model Training**: Compare DQN vs PPO performance
- **Curriculum Learning**: Progressive difficulty stages
- **Early Stopping**: Automatic training termination on performance thresholds

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

- **Gymnasium** for the modern RL environment framework (successor to OpenAI Gym)
- **Stable-Baselines3** for robust RL algorithm implementations
- **Optuna** for automated hyperparameter optimization
- **LoRaWAN Alliance** for IoT communication standards
- **Automotive safety research community** for domain expertise

## �️ Troubleshooting

### Common Issues and Solutions

#### "Gym has been unmaintained" Warning
- **Solution**: Already fixed! CrashGuard now uses Gymnasium
- **Command**: `pip install gymnasium>=0.28.0`

#### Model Uses Only One Action (e.g., 100% "Broadcast to Vehicles")
- **Cause**: Poor exploration or reward shaping
- **Solution**: Run enhanced training with longer exploration
```bash
python improve_accuracy.py  # Quick fix
# or
python train_enhanced.py --model DQN --balanced-training
```

#### Low Accuracy on Minor Crashes (0% accuracy)
- **Cause**: Imbalanced training data
- **Solution**: Use balanced training
```bash
python train.py --enhanced --balanced-training
```

#### Model File Not Found Error
- **Cause**: Incorrect model path or training not completed
- **Solution**: Check available models
```bash
# List available models
ls logs/*/
# Use correct path format
python evaluate.py --model-path logs/enhanced_training_*/best_model.zip
```

#### Training Crashes with Callback Error
- **Cause**: Incompatible callback configuration
- **Solution**: Use the fixed training scripts (already updated)

### Performance Optimization Tips

1. **For Higher Accuracy**: Use `--enhanced` flag with balanced training
2. **For Faster Training**: Reduce `--timesteps` and `--dataset-size`
3. **For Best Results**: Run hyperparameter optimization first
4. **For Stability**: Try PPO instead of DQN

## �📞 Support

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: [Create an issue](https://github.com/mritula2311/crash-guard/issues)
- **Documentation**: Check examples and inline documentation
- **Community**: Join discussions in GitHub Discussions

---

**CrashGuard** - Saving lives through intelligent crash detection and response 🚗🧠🚨