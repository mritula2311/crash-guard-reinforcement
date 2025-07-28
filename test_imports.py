#!/usr/bin/env python3
"""
Test script to check imports and basic functionality
"""

try:
    print("Testing imports...")
    
    print("1. Testing gymnasium...")
    import gymnasium as gym
    print("   ✓ gymnasium imported successfully")
    
    print("2. Testing stable-baselines3...")
    from stable_baselines3 import DQN, PPO
    print("   ✓ stable-baselines3 imported successfully")
    
    print("3. Testing other packages...")
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print("   ✓ numpy, pandas, matplotlib imported successfully")
    
    print("4. Testing os module...")
    import os
    print("   ✓ os imported successfully")
    
    print("5. Testing crash_guard imports...")
    from crash_guard import CrashGuardEnv, CrashGuardEvaluator
    print("   ✓ crash_guard imported successfully")
    
    print("6. Testing model loading...")
    model_path = "logs/training_dqn_20250728_062109/crash_guard_dqn_20250728_062111.zip"
    if os.path.exists(model_path):
        evaluator = CrashGuardEvaluator(model_path, 'DQN')
        print("   ✓ CrashGuardEvaluator created successfully")
    else:
        print(f"   ⚠ Model file not found: {model_path}")
    
    print("\nAll imports successful! ✅")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
