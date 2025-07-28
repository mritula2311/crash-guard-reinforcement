"""
CrashGuard: Reinforcement Learning Environment for Crash Severity Estimation
and Real-time Response Decision-making in IoT-based Vehicle Accident Detection.
"""

__version__ = "1.0.0"
__author__ = "CrashGuard Team"

from .environment import CrashGuardEnv
from .data_generator import CrashDataGenerator
from .trainer import CrashGuardTrainer
from .evaluator import CrashGuardEvaluator

__all__ = [
    "CrashGuardEnv",
    "CrashDataGenerator", 
    "CrashGuardTrainer",
    "CrashGuardEvaluator"
]